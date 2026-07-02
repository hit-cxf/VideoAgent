import json
import torch
from PIL import Image
import base64
import io
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import math
import tempfile
import sys
import re
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from environment.config.llm import gemini
from pydantic import BaseModel, Field
from environment.agents.base import BaseTool

EDITOR_VLM_CONCURRENCY = 50

class VideoEditor(BaseTool):
    """
    Agent that edit each retrieved video clips from video_dir based on timestamp file and, ultimately merging the fine-grained clips and adding audio.
    Prerequisites: Requires upstream agents to provide video frame timestamps (unless explicitly specified that user provides video frame timestamps.)
    Note: VideoPreloader, VideoSearcher, and VideoEditor need to be called together.
    """

    def __init__(self):
        super().__init__()
        # Navigate up to reach the root directory
        self.project_root = os.getcwd()

        # Define paths
        self.dataset_dir = os.path.join(self.project_root, 'dataset')
        self.video_edit_dir = os.path.join(self.dataset_dir, 'video_edit')
        self.audio_analysis_dir = os.path.join(self.video_edit_dir, 'audio_analysis')
        self.scene_output_dir = os.path.join(self.video_edit_dir, 'scene_output')
        self.working_dir = os.path.join(self.video_edit_dir, 'videosource-workdir')

        # Default paths
        self.ROOT_VIDEO_DIR = os.path.join(self.video_edit_dir, 'video_source')
        self.timestamp = None
        self.storyboard_file = os.path.join(self.scene_output_dir, "video_scene.json")
        self.audio_path = os.path.join(self.audio_analysis_dir, "gen_audio.wav")


        self.video_segments = []
        self.video_segments_data = {}

    class InputSchema(BaseTool.BaseInputSchema):
        video_dir: str = Field(
            ...,
            description="Directory containing source video files"
        )
        audio_path: str = Field(
            ...,
            description="Path to the audio"
        )
        timestamp_path: str = Field(
            ...,
            description="JSON File path used to store and load the timestamp of the end of each video segment, and determine when to switch to the next video segment based on the timestamp information. The purpose is to edit and splice videos"
        )

    class OutputSchema(BaseModel):
        video_path: str = Field(
            ...,
            description="Path to the generated video file"
        )

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for API"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _call_api(self, prompt: str, log_prefix: str = "") -> str:
        """Call API with retry logic"""
        prefix = f"{log_prefix} " if log_prefix else ""
        try:
            response = gemini(
                user=prompt
            )
            
            # Extract response content
            if hasattr(response, 'choices') and response.choices:
                response_text = response.choices[0].message.content.strip()
            else:
                response_text = str(response).strip()
            
            print(f"{prefix}API call successful")
            return response_text
            
        except Exception as e:
            print(f"{prefix}API call failed: {str(e)}")
            raise e

    def _analyze_frames(self, frames: List[Image.Image], description: str, exact_duration: float, log_prefix: str = "") -> int:
        """Use API to analyze frames and select the best starting frame"""
        prefix = f"{log_prefix} " if log_prefix else ""
        try:
            # Create multimodal content array
            content = [
                {
                    "type": "text",
                    "text": (
                        f"You are analyzing a video with {len(frames)} consecutive video frames, fps setting is 1. "
                        f"The required clip duration is {exact_duration:.3f} seconds.\n"
                        f"Find the best sequence matching this description:\n"
                        f"\"{description}\"\n\n"
                        f"Requirements:\n"
                        f"1. You must analyze ALL {len(frames)} frames to find the best consecutive sequence\n"
                        f"2. Choose a starting frame (0-{len(frames)-1}) that allows for a {exact_duration:.3f}s clip\n"
                        f"3. Maximum starting frame should be {len(frames) - math.ceil(exact_duration)} to fit the duration\n"
                        f"4. Return ONLY a single number - the starting frame number (0-{len(frames)-1})\n"
                        f"5. The returned consecutive sequence should align with the scene description\n"
                        f"6. Select consecutive frames with high-quality visuals and scene consistency\n\n"
                        f"7. Do not answer anything unrelated, return only exact one single number\n\n"
                        f"Here are the {len(frames)} frames for analysis:"
                    )
                }
            ]
            
            # Add each frame as an image input
            for i, frame in enumerate(frames):
                base64_image = self._encode_image_to_base64(frame)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_image
                    }
                })
            
            # Call Gemini API with multimodal content
            response = gemini(user=content)
            
            # Extract response content
            if hasattr(response, 'choices') and response.choices:
                response_text = response.choices[0].message.content.strip()
            else:
                response_text = str(response).strip()
            
            print(f"{prefix}API call successful")
            
            # Extract frame number from response
            frame_numbers = re.findall(r'\d+', response_text)
            if not frame_numbers:
                print(f"{prefix}No frame number found in response, using 0")
                return 0
                
            frame_number = int(frame_numbers[0])
            max_start_frame = len(frames) - math.ceil(exact_duration)
            
            if 0 <= frame_number <= max_start_frame:
                print(f"{prefix}Selected frame {frame_number} from response")
                return frame_number
            else:
                print(f"{prefix}Frame number {frame_number} out of bounds (max: {max_start_frame}), using 0")
                return 0
                
        except Exception as e:
            print(f"{prefix}Error in Gemini analysis: {str(e)}")
            print(f"{prefix}Falling back to frame 0")
            return 0  # Fallback to first frame

    def _job_log_prefix(self, job: Dict) -> str:
        return (
            f"[Period {job['index'] + 1}]"
            f"[{job['segment_name']}]"
            f"[BGM {job['period_start']:.3f}-{job['period_end']:.3f}s]"
            f"[SRC {job['segment_start']:.3f}-{job['segment_end']:.3f}s]"
        )

    def _load_data(self):
        """Load video segments and timing data"""
        # Load segment data
        segments_path = os.path.join(self.scene_output_dir, 'visual_retrieved_segments.json')
        if os.path.exists(segments_path):
            with open(segments_path, 'r', encoding='utf-8') as f:
                self.video_segments = json.load(f)
                print(f"Loaded {len(self.video_segments)} video segments from: {segments_path}")
        else:
            print(f"Warning: {segments_path} not found. Using default empty list.")
            self.video_segments = []

        kv_store_path = os.path.join(self.working_dir, 'kv_store_video_segments.json')
        if os.path.exists(kv_store_path):
            with open(kv_store_path, 'r', encoding='utf-8') as f:
                self.video_segments_data = json.load(f)
                print(f"Loaded timing data from: {kv_store_path}")
        else:
            print(f"Warning: {kv_store_path} not found. Using default empty dict.")
            self.video_segments_data = {}

    def _load_video_timing(self, segment_name: str):
        """Load video timing from segment name"""
        try:
            parts = segment_name.split('_')

            # Handle different format cases
            if len(parts) == 3:
                # Format: movie_video_id_section (e.g., "movie_0_105")
                movie = parts[0]
                video_id = parts[1]
                section = parts[2]
                video_key = f"{movie}_{video_id}"
            elif len(parts) == 2:
                # Format: movie_section (e.g., "movie1_105")
                movie = parts[0]
                section = parts[1]
                video_key = movie
            else:
                print(f"Invalid segment name format: {segment_name}")
                return None, None

            # Check if data exists and retrieve timing
            if video_key in self.video_segments_data and section in self.video_segments_data[video_key]:
                timing = self.video_segments_data[video_key][section]['time']
                start_time, end_time = map(float, timing.split('-'))
                return start_time, end_time
            else:
                print(f"No timing data found for {segment_name} (key: {video_key}, section: {section})")
                return None, None
        except Exception as e:
            print(f"Error processing timing for {segment_name}: {str(e)}")
            return None, None

    def _get_video_path(self, segment_name: str) -> str:
        """Get video file path from segment name"""
        try:
            parts = segment_name.split('_')

            if len(parts) == 3:
                # Format: movie_video_id_section
                movie_name = parts[0]
                video_num = parts[1]
                video_filename = f"{movie_name}_{video_num}.mp4"
            elif len(parts) == 2:
                # Format: movie_section
                movie_name = parts[0]
                video_filename = f"{movie_name}.mp4"
            else:
                print(f"Invalid segment name format for video path: {segment_name}")
                return None

            video_path = os.path.join(self.ROOT_VIDEO_DIR, video_filename)
            if not os.path.exists(video_path):
                print(f"Video file not found at: {video_path}")
            return video_path
        except Exception as e:
            print(f"Error getting video path for {segment_name}: {str(e)}")
            return None

    def _extract_frames(self, video: VideoFileClip, start_time: float, end_time: float) -> List[Tuple[float, Image.Image]]:
        """Extract frames including the exact start time"""
        frames = []
        try:
            # Start from exact start_time, then continue with whole seconds
            frames_times = [start_time]  # Include exact start time
            current_time = math.ceil(start_time)  # Round up to next second

            while current_time < end_time:
                frames_times.append(current_time)
                current_time += 1

            for t in frames_times:
                try:
                    if t >= end_time:
                        break
                    frame = video.get_frame(t)
                    # Convert to RGB if necessary
                    if frame.shape[2] == 4:  # If RGBA
                        frame = Image.fromarray(frame).convert('RGB')
                    else:
                        frame = Image.fromarray(frame)

                    # Resize with consistent dimensions
                    frame = frame.resize((224, 224), Image.Resampling.LANCZOS)
                    frames.append((t, frame))
                except Exception as e:
                    print(f"Error extracting frame at time {t:.3f}: {e}")
                    continue

            return frames
        except Exception as e:
            print(f"Error in frame extraction: {e}")
            return frames

    def _process_video(self, beats_file: str, storyboard_file: str, bgm_file: str, keep_original_audio: bool = False, audio_mix_ratio: float = 0.3, output_path: str = "output_video.mp4"):
        """Main video processing pipeline"""
        import re
        import math
        final_clips = []
        final_video = None
        background_audio = None
        total_duration = 0

        try:
            # Ensure file paths are absolute
            beats_file = os.path.join(self.working_dir, beats_file) if not os.path.isabs(beats_file) else beats_file
            storyboard_file = os.path.join(self.working_dir, storyboard_file) if not os.path.isabs(storyboard_file) else storyboard_file

            print(f"Processing with files:")
            print(f"  Beats: {beats_file}")
            print(f"  Storyboard: {storyboard_file}")
            print(f"  Audio: {bgm_file}")
            print(f"  Output: {output_path}")

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Load beat timestamps for time periods
            with open(beats_file, 'r', encoding='utf-8') as f:
                beats_data = json.load(f)

            # Handle different data formats
            if 'beat_data' in beats_data and 'beats' in beats_data['beat_data']:
                beat_timestamps = [beat['timestamp'] for beat in beats_data['beat_data']['beats']]
            elif 'sentence_data' in beats_data and 'chunks' in beats_data['sentence_data']:
                beat_timestamps = [cut['timestamp'] for cut in beats_data['sentence_data']['chunks']]
            else:
                raise ValueError("Unrecognized timestamp data format")

            print(f"Found {len(beat_timestamps)} timestamps")
            print(f"First 5 timestamps: {beat_timestamps[:5]}")

            # Create time periods starting from 0
            time_periods = [(0, beat_timestamps[0])] if beat_timestamps else []
            time_periods.extend([(beat_timestamps[i], beat_timestamps[i+1])
                            for i in range(len(beat_timestamps)-1)])

            print(f"Created {len(time_periods)} time periods")

            # Load storyboard
            with open(storyboard_file, 'r', encoding='utf-8') as f:
                storyboard_data = json.load(f)

            storyboard_text = storyboard_data.get('segment_scene', '')

            storyboard_sections = [section.strip()
                                for section in storyboard_text.split('/////')
                                if section.strip()]

            print(f"Total video segments: {len(self.video_segments)}")
            print(f"Total storyboard sections: {len(storyboard_sections)}")

            max_periods = min(len(time_periods), len(storyboard_sections))
            print(f"Will process {max_periods} periods (limited by storyboard sections)")

            analysis_jobs = []

            # Collect frames and metadata for each time period. VLM frame selection
            # is batched below; video clipping remains ordered and serial.
            for j in range(max_periods):
                period_start, period_end = time_periods[j]
                exact_duration = period_end - period_start
                period_prefix = f"[Period {j + 1}][BGM {period_start:.3f}-{period_end:.3f}s]"
                print(f"\n{period_prefix} Processing time period (Duration: {exact_duration:.3f}s)")

                if not self.video_segments:
                    print(f"{period_prefix} No video segments available")
                    break

                segment_idx = j % len(self.video_segments)
                segment_name = self.video_segments[segment_idx]

                try:
                    segment_start, segment_end = self._load_video_timing(segment_name)
                    segment_prefix = (
                        f"[Period {j + 1}]"
                        f"[{segment_name}]"
                        f"[BGM {period_start:.3f}-{period_end:.3f}s]"
                        f"[SRC {segment_start:.3f}-{segment_end:.3f}s]"
                    ) if segment_start is not None and segment_end is not None else f"{period_prefix}[{segment_name}]"

                    if segment_start is None or segment_end is None:
                        print(f"{segment_prefix} Skipping segment - invalid timing")
                        continue

                    print(f"{segment_prefix} Checking segment")
                    print(f"{segment_prefix} Segment range ready")

                    # Validate segment duration
                    max_start = segment_end - exact_duration
                    if max_start < segment_start:
                        print(f"{segment_prefix} Segment too short for required duration {exact_duration:.3f}s")
                        continue

                    if j >= len(storyboard_sections):
                        print(f"{segment_prefix} Not enough storyboard sections")
                        break

                    current_section = storyboard_sections[j]
                    description = '\n'.join(current_section.split('\n')[1:]).strip()
                    print(f"{segment_prefix} Using storyboard section {j + 1}")

                    video_path = self._get_video_path(segment_name)
                    if not os.path.exists(video_path):
                        print(f"{segment_prefix} Video file not found: {video_path}")
                        continue

                    # Load with audio if we're keeping original audio
                    with VideoFileClip(video_path, audio=keep_original_audio) as temp_video:
                        frames_with_times = self._extract_frames(temp_video, segment_start, segment_end)

                    if not frames_with_times:
                        print(f"{segment_prefix} No frames extracted")
                        continue

                    print(f"{segment_prefix} Extracted {len(frames_with_times)} frames for analysis")

                    timestamps = [t for t, _ in frames_with_times]
                    frames = [f for _, f in frames_with_times]

                    analysis_jobs.append({
                        "index": j,
                        "segment_name": segment_name,
                        "video_path": video_path,
                        "period_start": period_start,
                        "period_end": period_end,
                        "segment_start": segment_start,
                        "segment_end": segment_end,
                        "exact_duration": exact_duration,
                        "description": description,
                        "frames": frames,
                        "frame_count": len(frames),
                    })

                except Exception as e:
                    print(f"{period_prefix}[{segment_name}] Error preparing segment: {e}")
                    continue

            if not analysis_jobs:
                print("No valid frame-analysis jobs to process")
                return {
                    "output_path": ""
                }

            print(f"\nAnalyzing {len(analysis_jobs)} frame jobs with Gemini (concurrency={EDITOR_VLM_CONCURRENCY})...")

            def run_analysis(job):
                log_prefix = self._job_log_prefix(job)
                print(f"{log_prefix} FrameAnalysis start")
                frame_number = self._analyze_frames(
                    job["frames"],
                    job["description"],
                    job["exact_duration"],
                    log_prefix=log_prefix,
                )
                print(f"{log_prefix} FrameAnalysis selected frame {frame_number}")
                return job["index"], frame_number

            frame_results = {}
            with ThreadPoolExecutor(max_workers=EDITOR_VLM_CONCURRENCY) as executor:
                futures = [executor.submit(run_analysis, job) for job in analysis_jobs]
                for future in as_completed(futures):
                    try:
                        job_index, frame_number = future.result()
                        frame_results[job_index] = frame_number
                    except Exception as e:
                        print(f"[FrameAnalysis] Unexpected error: {e}")

            # Create clips in original timeline order.
            for job in sorted(analysis_jobs, key=lambda item: item["index"]):
                try:
                    j = job["index"]
                    frame_number = frame_results.get(j, 0)
                    segment_start = job["segment_start"]
                    exact_duration = job["exact_duration"]
                    video_path = job["video_path"]
                    frame_count = job["frame_count"]
                    log_prefix = self._job_log_prefix(job)

                    clip_start = segment_start + frame_number  # Start from frame_number seconds into segment
                    clip_end = clip_start + exact_duration
                    print(f"{log_prefix} Selected clip: starting from frame {frame_number} of {frame_count - 1}")
                    print(f"{log_prefix} Precise timing: {clip_start:.3f}s - {clip_end:.3f}s")

                    # Create clip with precise timing - now loading with audio based on keep_original_audio
                    clip = VideoFileClip(video_path, audio=keep_original_audio).subclip(clip_start, clip_end)
                    final_clips.append(clip)

                    total_duration += clip.duration
                    print(f"{log_prefix} Added clip: Duration = {clip.duration:.3f}s")
                    print(f"{log_prefix} Current total duration: {total_duration:.3f}s")

                except Exception as e:
                    print(f"{self._job_log_prefix(job)} Error creating clip: {e}")
                    continue

            if not final_clips:
                print("No valid clips to process")
                return {
                    "output_path": ""
                }

            print(f"\nConcatenating {len(final_clips)} clips...")
            final_video = concatenate_videoclips(final_clips, method="compose")

            # Audio handling based on the keep_original_audio option
            if keep_original_audio:
                # If keeping original audio, mix it with the background music
                print("Loading background music...")
                background_audio = AudioFileClip(bgm_file).subclip(0, final_video.duration)

                if audio_mix_ratio > 0:
                    print(f"Mixing original audio with background music (ratio: {audio_mix_ratio:.2f})")
                    # Adjust background volume (background is quieter)
                    background_audio = background_audio.volumex(audio_mix_ratio)

                    # Combine original audio with background music
                    mixed_audio = CompositeAudioClip([
                        final_video.audio,  # Original audio
                        background_audio    # Background music at reduced volume
                    ])
                    final_video = final_video.set_audio(mixed_audio)
                else:
                    # Keep only original audio, ignore background music
                    print("Using only original audio (no background music)")
            else:
                # No original audio, just use background music
                print("Adding background music only...")
                background_audio = AudioFileClip(bgm_file)
                background_audio = background_audio.subclip(0, final_video.duration)
                final_video = final_video.set_audio(background_audio)

            print(f"Writing final video to {output_path}...")
            final_video.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                threads=4,
                preset='medium',
                temp_audiofile=None,
                remove_temp=True,
                verbose=False
            )
            print("Video processing completed!")

            return {
                "video_path": output_path
            }

        except Exception as e:
            print(f"Error in video processing: {e}")
            import traceback
            traceback.print_exc()
            return {
                "video_path": ""
            }

        finally:
            try:
                # Clean up resources
                for clip in final_clips:
                    clip.close()
                if final_video is not None:
                    final_video.close()
                if background_audio is not None:
                    background_audio.close()
            except:
                pass

    def execute(self, **kwargs):
        """Main method to execute the video creation process"""
        # Validate input parameters
        params = self.InputSchema(**kwargs)

        self.timestamp = params.timestamp_path
        keep_original_audio = False
        audio_mix_ratio = 0.3

        # Update the root video directory if provided
        if params.video_dir:
            # Handle the case where input_path might be a string with parentheses
            if isinstance(params.video_dir, str) and '(' in params.video_dir and ')' in params.video_dir:
                # Extract the path from parentheses if needed
                input_path = params.video_dir.strip('()')
            else:
                input_path = params.video_dir

            self.ROOT_VIDEO_DIR = input_path
            print(f"Using custom video directory: {self.ROOT_VIDEO_DIR}")

        # Load necessary data
        self._load_data()

        # Use custom audio file if provided
        synth_audio_path = params.audio_path
        print(f"Using bgm file: {synth_audio_path}")

        # Set output file path
        output_path = 'dataset/final.mp4'

        # Verify files exist
        for file_path in [self.timestamp, self.storyboard_file, synth_audio_path]:
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                if file_path == self.timestamp or file_path == self.storyboard_file:
                    return {
                        "output_path": "",
                        "duration": 0,
                        "clip_count": 0,
                        "status": f"error: required file not found: {file_path}"
                    }

        # Process the video
        result = self._process_video(
            beats_file=self.timestamp,
            storyboard_file=self.storyboard_file,
            bgm_file=synth_audio_path,
            keep_original_audio=keep_original_audio,
            audio_mix_ratio=audio_mix_ratio,
            output_path=output_path
        )

        return result
