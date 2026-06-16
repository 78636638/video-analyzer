import argparse
from datetime import datetime
from pathlib import Path
import json
import logging
import shutil
import sys
import time
import urllib.request
from typing import Optional

from .config import Config, get_client, get_client_by_type, get_model, get_model_by_type
from .frame import VideoProcessor
from .prompt import PromptLoader
from .analyzer import VideoAnalyzer
from .audio_processor import AudioProcessor
from .clients.ollama import OllamaClient
from .clients.generic_openai_api import GenericOpenAIAPIClient

# Initialize logger at module level
logger = logging.getLogger(__name__)

# #region debug-point A:cli-stage-metrics
def _debug_emit(hypothesis_id: str, location: str, msg: str, data: Optional[dict] = None, run_id: str = "pre") -> None:
    env_path = Path(".dbg/long-video-performance.env")
    server_url = "http://127.0.0.1:7777/event"
    session_id = "long-video-performance"
    if env_path.exists():
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DEBUG_SERVER_URL="):
                    server_url = line.split("=", 1)[1].strip()
                elif line.startswith("DEBUG_SESSION_ID="):
                    session_id = line.split("=", 1)[1].strip()
        except Exception:
            return

    payload = {
        "sessionId": session_id,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "msg": f"[DEBUG] {msg}",
        "data": data or {},
        "ts": int(time.time() * 1000),
    }
    try:
        urllib.request.urlopen(
            urllib.request.Request(
                server_url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=1,
        ).read()
    except Exception:
        pass
# #endregion

def get_log_level(level_str: str) -> int:
    """Convert string log level to logging constant."""
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)

def cleanup_files(output_dir: Path):
    """Clean up temporary files and directories."""
    try:
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
            logger.debug(f"Cleaned up frames directory: {frames_dir}")
            
        audio_file = output_dir / "audio.wav"
        if audio_file.exists():
            audio_file.unlink()
            logger.debug(f"Cleaned up audio file: {audio_file}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def prepare_output_dir(output_dir: Path, start_stage: int) -> None:
    """Back up the existing output directory before starting a fresh task."""
    if start_stage > 1:
        logger.info(
            "Skipping output backup because start_stage=%s expects existing artifacts.",
            start_stage,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    if output_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backups_root = output_dir.parent / f"{output_dir.name}_backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        backup_name = f"{output_dir.name}_{timestamp}"
        backup_dir = backups_root / backup_name
        counter = 1
        while backup_dir.exists():
            backup_dir = backups_root / f"{backup_name}_{counter}"
            counter += 1

        shutil.move(str(output_dir), str(backup_dir))
        logger.info("Backed up existing output directory to %s", backup_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Prepared fresh output directory at %s", output_dir)

def create_client(config: Config):
    """Create the appropriate client based on configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    return create_client_by_type(config, client_type)


def create_client_by_type(config: Config, client_type: str):
    """Create the appropriate client for a specific configuration type."""
    client_config = get_client(config)
    if client_type != config.get("clients", {}).get("default", "ollama"):
        client_config = get_client_by_type(config, client_type)

    if client_type == "ollama":
        return OllamaClient(client_config["url"])
    elif client_type == "openai_api":
        return GenericOpenAIAPIClient(client_config["api_key"], client_config["api_url"])
    else:
        raise ValueError(f"Unknown client type: {client_type}")

def main():
    parser = argparse.ArgumentParser(description="Analyze video using Vision models")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--config", type=str, default="config",
                        help="Path to configuration directory")
    parser.add_argument("--output", type=str, help="Output directory for analysis results")
    parser.add_argument("--client", type=str, help="Client to use (ollama or openrouter)")
    parser.add_argument("--ollama-url", type=str, help="URL for the Ollama service")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI-compatible service")
    parser.add_argument("--api-url", type=str, help="API URL for OpenAI-compatible API")
    parser.add_argument("--model", type=str, help="Name of the vision model to use")
    parser.add_argument("--duration", type=float, help="Duration in seconds to process")
    parser.add_argument("--keep-frames", action="store_true", help="Keep extracted frames after analysis")
    parser.add_argument("--whisper-model", type=str, help="Whisper model size (tiny, base, small, medium, large), or path to local Whisper model snapshot")
    parser.add_argument("--start-stage", type=int, default=1, help="Stage to start processing from (1-3)")
    parser.add_argument("--max-frames", type=int, default=sys.maxsize, help="Maximum number of frames to process")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    parser.add_argument("--prompt", type=str, default="",
                        help="Question to ask about the video")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temperature", type=float, help="Temperature for LLM generation")
    args = parser.parse_args()

    # Set up logging with specified level
    log_level = get_log_level(args.log_level)
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # Force reconfiguration of the root logger
    )
    # Ensure our module logger has the correct level
    logger.setLevel(log_level)

    main_started_at = time.perf_counter()

    # Load and update configuration
    config = Config(args.config)
    config.update_from_args(args)

    # Initialize components
    video_path = Path(args.video_path)
    output_dir = Path(config.get("output_dir"))
    prepare_output_dir(output_dir, args.start_stage)
    client = create_client(config)
    model = get_model(config)
    prompt_loader = PromptLoader(config.get("prompt_dir"), config.get("prompts", []))
    analysis_config = config.get("analysis", {}) or {}
    verifier_enabled = bool(analysis_config.get("stage_verifier_enabled", False))
    verifier_client_type = str(analysis_config.get("stage_verifier_client", "")).strip()
    verifier_model = str(analysis_config.get("stage_verifier_model", "")).strip()
    verifier_client = None
    if verifier_enabled and verifier_client_type:
        verifier_client = create_client_by_type(config, verifier_client_type)
        if not verifier_model:
            verifier_model = get_model_by_type(config, verifier_client_type)
    _debug_emit(
        "A",
        "cli.py:main",
        "CLI analysis started",
        {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "duration_arg": args.duration,
            "max_frames_arg": args.max_frames,
            "client": config.get("clients", {}).get("default"),
            "model": model,
            "stage_verifier_enabled": verifier_enabled,
            "stage_verifier_client": verifier_client_type or None,
            "stage_verifier_model": verifier_model or None,
            "analysis_config": analysis_config,
        },
    )
    if hasattr(client, "configure_image_encoding"):
        client.configure_image_encoding(
            max_image_side=analysis_config.get("max_image_side"),
            jpeg_quality=analysis_config.get("image_jpeg_quality", 85),
        )
    if verifier_client and hasattr(verifier_client, "configure_image_encoding"):
        verifier_client.configure_image_encoding(
            max_image_side=analysis_config.get("max_image_side"),
            jpeg_quality=analysis_config.get("image_jpeg_quality", 85),
        )
    
    try:
        transcript = None
        audio_summary = None
        audio_memory = []
        frames = []
        frame_analyses = []
        video_description = None
        analyzer = None
        
        # Stage 1: Frame and Audio Processing
        if args.start_stage <= 1:
            stage1_started_at = time.perf_counter()
            # Initialize audio processor and extract transcript, the AudioProcessor accept following parameters that can be set in config.json:
            # language (str): Language code for audio transcription (default: None)
            # whisper_model (str): Whisper model size or path (default: "medium")
            # device (str): Device to use for audio processing (default: "cpu")
            logger.debug("Initializing audio processing...")
            audio_processor = AudioProcessor(language=config.get("audio", {}).get("language", ""), 
                                             model_size_or_path=config.get("audio", {}).get("whisper_model", "medium"),
                                             device=config.get("audio", {}).get("device", "cpu"))
            
            logger.info("Extracting audio from video...")
            try:
                audio_path = audio_processor.extract_audio(video_path, output_dir)
            except Exception as e:
                logger.error(f"Error extracting audio: {e}")
                audio_path = None
            
            if audio_path is None:
                logger.debug("No audio found in video - skipping transcription")
                transcript = None
                _debug_emit("A", "cli.py:stage1", "Audio extraction finished without track", {})
            else:
                logger.info("Transcribing audio...")
                transcribe_started_at = time.perf_counter()
                transcript = audio_processor.transcribe(audio_path)
                _debug_emit(
                    "A",
                    "cli.py:stage1",
                    "Audio transcription finished",
                    {
                        "elapsed_seconds": round(time.perf_counter() - transcribe_started_at, 3),
                        "transcript_available": transcript is not None,
                        "segments_count": len(transcript.segments) if transcript else 0,
                    },
                )
                if transcript is None:
                    logger.warning("Could not generate reliable transcript. Proceeding with video analysis only.")
                else:
                    audio_summary = audio_processor.summarize_audio_segments(
                        transcript,
                        chunk_length=config.get("audio", {}).get("chunk_length", 30),
                    )
                    audio_memory = audio_processor.build_audio_snippets(
                        transcript,
                        alignment_window=analysis_config.get("audio_alignment_window", 2.0),
                    )
            
            logger.info(f"Extracting frames from video using model {model}...")
            processor = VideoProcessor(
                video_path, 
                output_dir / "frames", 
                model
            )
            frames = processor.extract_keyframes(
                frames_per_minute=config.get("frames", {}).get("per_minute", 60),
                duration=config.get("duration"),
                max_frames=args.max_frames,
                analysis_config=analysis_config,
            )
            _debug_emit(
                "D",
                "cli.py:stage1",
                "Stage 1 completed",
                {
                    "elapsed_seconds": round(time.perf_counter() - stage1_started_at, 3),
                    "audio_available": transcript is not None,
                    "audio_memory_count": len(audio_memory),
                    "frames_extracted": len(frames),
                },
            )
            
        # Stage 2: Frame Analysis
        if args.start_stage <= 2:
            logger.info("Analyzing frames...")
            stage2_started_at = time.perf_counter()
            analyzer = VideoAnalyzer(
                client, 
                model, 
                prompt_loader,
                config.get("clients", {}).get("temperature", 0.2),
                config.get("prompt", ""),
                analysis_config=analysis_config,
                verifier_client=verifier_client,
                verifier_model=verifier_model,
            )
            analyzer.set_audio_memory(audio_memory)
            analyzer.set_audio_summary(audio_summary)
            frame_analyses = []
            total_frames = len(frames)
            if analyzer.enable_stage_batch_analysis:
                batch_size = analyzer.stage_batch_size
                total_stages = (total_frames + batch_size - 1) // batch_size
                processed_frames = 0
                for stage_index in range(total_stages):
                    batch_frames = frames[stage_index * batch_size : (stage_index + 1) * batch_size]
                    stage_started_at = time.perf_counter()
                    stage_results = analyzer.analyze_frame_batch(
                        batch_frames,
                        stage_id=stage_index + 1,
                        total_stages=total_stages,
                    )
                    frame_analyses.extend(stage_results)
                    processed_frames += len(batch_frames)
                    _debug_emit(
                        "A",
                        "cli.py:stage2",
                        "Batch stage analysis completed",
                        {
                            "stage_index": stage_index + 1,
                            "total_stages": total_stages,
                            "frame_range": [batch_frames[0].number, batch_frames[-1].number],
                            "frames_in_stage": len(batch_frames),
                            "elapsed_seconds": round(time.perf_counter() - stage_started_at, 3),
                            "response_length": len((stage_results[0] or {}).get("response", "")) if stage_results else 0,
                        },
                    )
                    logger.info(
                        "Frame analysis progress: %s/%s (stage=%s/%s frames=%s-%s)",
                        processed_frames,
                        total_frames,
                        stage_index + 1,
                        total_stages,
                        batch_frames[0].number,
                        batch_frames[-1].number,
                    )
            else:
                for index, frame in enumerate(frames, 1):
                    frame_started_at = time.perf_counter()
                    analysis = analyzer.analyze_frame(frame)
                    frame_analyses.append(analysis)
                    _debug_emit(
                        "A",
                        "cli.py:stage2",
                        "Frame analysis completed",
                        {
                            "frame_index": index,
                            "total_frames": total_frames,
                            "frame_number": frame.number,
                            "timestamp": round(frame.timestamp, 3),
                            "elapsed_seconds": round(time.perf_counter() - frame_started_at, 3),
                            "response_length": len((analysis or {}).get("response", "")),
                        },
                    )
                    logger.info(
                        "Frame analysis progress: %s/%s (frame=%s timestamp=%.2fs)",
                        index,
                        total_frames,
                        frame.number,
                        frame.timestamp,
                    )
            _debug_emit(
                "A",
                "cli.py:stage2",
                "Stage 2 completed",
                {
                    "elapsed_seconds": round(time.perf_counter() - stage2_started_at, 3),
                    "frames_processed": len(frame_analyses),
                },
            )
                
        # Stage 3: Video Reconstruction
        if args.start_stage <= 3:
            stage3_started_at = time.perf_counter()
            if analyzer is None:
                analyzer = VideoAnalyzer(
                    client,
                    model,
                    prompt_loader,
                    config.get("clients", {}).get("temperature", 0.2),
                    config.get("prompt", ""),
                    analysis_config=analysis_config,
                )
                analyzer.set_audio_memory(audio_memory)
                analyzer.set_audio_summary(audio_summary)
            logger.info("Reconstructing video description...")
            video_description = analyzer.reconstruct_video(
                frame_analyses, frames, transcript
            )
            _debug_emit(
                "C",
                "cli.py:stage3",
                "Stage 3 completed",
                {
                    "elapsed_seconds": round(time.perf_counter() - stage3_started_at, 3),
                    "chunk_summaries": len((video_description or {}).get("chunk_summaries", [])),
                    "scene_cards": len((video_description or {}).get("scene_cards", [])),
                    "story_beats": len((video_description or {}).get("story_beats", [])),
                    "character_timeline": len((video_description or {}).get("character_timeline", [])),
                    "video_description_length": len((video_description or {}).get("response", "")),
                },
            )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "metadata": {
                "client": config.get("clients", {}).get("default"),
                "model": model,
                "whisper_model": config.get("audio", {}).get("whisper_model"),
                "frames_per_minute": config.get("frames", {}).get("per_minute"),
                "duration_processed": config.get("duration"),
                "frames_extracted": len(frames),
                "frames_processed": min(len(frames), args.max_frames),
                "start_stage": args.start_stage,
                "audio_language": transcript.language if transcript else None,
                "transcription_successful": transcript is not None,
                "analysis": analysis_config,
            },
            "transcript": {
                "text": transcript.text if transcript else None,
                "segments": transcript.segments if transcript else None
            } if transcript else None,
            "audio_summary": audio_summary,
            "frame_analyses": frame_analyses,
            "stage_batch_analyses": video_description.get("stage_batch_analyses", []) if video_description else [],
            "chunk_summaries": video_description.get("chunk_summaries", []) if video_description else [],
            "story_memory": video_description.get("story_memory", {}) if video_description else {},
            "scene_cards": video_description.get("scene_cards", []) if video_description else [],
            "story_beats": video_description.get("story_beats", []) if video_description else [],
            "character_timeline": video_description.get("character_timeline", []) if video_description else [],
            "script_guidance": video_description.get("script_guidance", {}) if video_description else {},
            "video_script": video_description.get("video_script", {}) if video_description else {},
            "video_description": {
                "response": video_description.get("response", "No description generated")
            } if video_description else None,
        }
        
        with open(output_dir / "analysis.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("\nTranscript:")
        if transcript:
            logger.info(transcript.text)
        else:
            logger.info("No reliable transcript available")
            
        if video_description:
            logger.info("\nVideo Description:")
            logger.info(video_description.get("response", "No description generated"))
        
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
        
        _debug_emit(
            "A",
            "cli.py:main",
            "CLI analysis completed",
            {
                "total_elapsed_seconds": round(time.perf_counter() - main_started_at, 3),
                "output_file": str(output_dir / "analysis.json"),
            },
        )
        logger.info(f"Analysis complete. Results saved to {output_dir / 'analysis.json'}")
            
    except Exception as e:
        _debug_emit("A", "cli.py:main", "CLI analysis failed", {"error": str(e)})
        logger.error(f"Error during video analysis: {e}")
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
        raise

if __name__ == "__main__":
    main()
