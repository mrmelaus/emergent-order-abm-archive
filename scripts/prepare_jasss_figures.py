#!/usr/bin/env python3
"""
prepare_jasss_figures.py
Purpose: Validate and process figures for JASSS submission, using the
         project's figure manifest (figure, source_subfolder, source_stem)
         to trace each Figure_N back to its original analysis output,
         copy it into the submission folder, and enforce JASSS's actual
         figure requirement (<=800px wide) rather than file size or DPI,
         which JASSS does not specify anywhere in their submission
         instructions.

Usage: python3 scripts/prepare_jasss_figures.py
"""

import os
import sys
import shutil
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

try:
    from PIL import Image
except ImportError:
    print("PIL (Pillow) not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths — resolved relative to this script's own location, so the
    # archive runs correctly regardless of where it's extracted/cloned.
    _THIS_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _THIS_DIR.parent
    BASE_PATH = _PROJECT_ROOT / "analysis" / "figures"
    SUBMISSION_PATH = BASE_PATH / "submission"
    BACKUP_PATH = SUBMISSION_PATH / "backup"
    MANIFEST_PATH = SUBMISSION_PATH / "submission_manifest.csv"

    # JASSS's ACTUAL figure requirement (confirmed from their submission
    # instructions page): figures must be no more than 800px wide. Height
    # of ~400-800px is stated as typical/soft guidance, not a hard rule,
    # so it is reported but never causes a resize or a failure.
    # JASSS specifies no file-size or DPI requirement anywhere.
    MAX_WIDTH_PX = 800
    SOFT_MIN_HEIGHT_PX = 400
    SOFT_MAX_HEIGHT_PX = 800

    PREFERRED_FORMATS = ['.png', '.pdf']  # PNG preferred per JASSS; PDF
                                            # kept alongside as archival/
                                            # LaTeX-route source, not itself
                                            # a submitted figure format


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_message(level: str, message: str):
    print(f"[{get_timestamp()}] {level}: {message}")


# ============================================================================
# MANIFEST LOADING
# ============================================================================

def load_manifest(manifest_path: Path) -> List[Dict]:
    """
    Load figure manifest from CSV file.
    Expected format: figure, source_subfolder, source_stem
    """
    manifest = []

    if not manifest_path.exists():
        log_message("ERROR", f"Manifest file not found: {manifest_path}")
        return manifest

    try:
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                figure = row.get('figure', '').strip()
                subfolder = row.get('source_subfolder', '').strip()
                stem = row.get('source_stem', '').strip()

                if not all([figure, subfolder, stem]):
                    log_message("WARNING", f"Skipping invalid row: {row}")
                    continue

                manifest.append({
                    'figure': figure,
                    'source_subfolder': subfolder,
                    'source_stem': stem
                })

        log_message("INFO", f"Loaded {len(manifest)} figures from manifest")
        return manifest

    except Exception as e:
        log_message("ERROR", f"Failed to load manifest: {e}")
        return []


# ============================================================================
# FILE PROCESSING FUNCTIONS
# ============================================================================

def get_file_info(file_path: Path) -> Dict:
    """Get detailed file information."""
    info = {
        'path': str(file_path),
        'name': file_path.name,
        'size_bytes': file_path.stat().st_size,
        'size_mb': file_path.stat().st_size / (1024 * 1024),
        'extension': file_path.suffix.lower(),
        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        'exists': file_path.exists()
    }

    if file_path.suffix.lower() == '.png':
        try:
            with Image.open(file_path) as img:
                info['width'] = img.width
                info['height'] = img.height
                info['mode'] = img.mode
                info['format'] = img.format
        except Exception as e:
            info['image_error'] = str(e)

    return info


def validate_against_jasss(file_info: Dict, config: Config) -> Tuple[bool, List[str]]:
    """
    Validate a PNG against JASSS's actual stated requirement: width
    <=800px. Height outside the 400-800px soft range is reported as an
    informational note, not a failure -- JASSS's own wording ("typically
    about 600 to 800 pixels in height") is guidance, not a hard rule,
    and wide/short comparison charts routinely and acceptably fall
    outside it.
    """
    issues = []
    is_valid = True

    if file_info['extension'] != '.png':
        return True, issues  # PDFs are not subject to this pixel check

    if 'width' not in file_info:
        issues.append("Could not read image dimensions")
        return False, issues

    width = file_info['width']
    height = file_info['height']

    if width > config.MAX_WIDTH_PX:
        is_valid = False
        issues.append(f"Width {width}px exceeds JASSS's {config.MAX_WIDTH_PX}px limit")

    if not (config.SOFT_MIN_HEIGHT_PX <= height <= config.SOFT_MAX_HEIGHT_PX):
        # Informational only -- does not affect is_valid
        issues.append(
            f"Height {height}px is outside JASSS's soft {config.SOFT_MIN_HEIGHT_PX}-"
            f"{config.SOFT_MAX_HEIGHT_PX}px guidance (not a hard requirement; "
            f"common and acceptable for wide/short comparison charts)"
        )

    return is_valid, issues


def resize_to_max_width(input_path: Path, output_path: Path, max_width: int) -> bool:
    """
    Resize a PNG so its width does not exceed max_width, preserving
    aspect ratio. Uses LANCZOS resampling for quality. Does nothing
    (just copies) if already within the limit.
    """
    try:
        with Image.open(input_path) as img:
            if img.width <= max_width:
                shutil.copy2(input_path, output_path)
                return True

            scale = max_width / img.width
            new_size = (max_width, round(img.height * scale))
            resized = img.resize(new_size, Image.LANCZOS)
            resized.save(output_path, 'PNG', optimize=True)
            return True

    except Exception as e:
        log_message("ERROR", f"Resize failed for {input_path.name}: {e}")
        return False


def locate_source_file(base_path: Path, subfolder: str, stem: str) -> Dict:
    """Locate source file(s) for a figure."""
    source_dir = base_path / subfolder
    result = {'found': False, 'png': None, 'pdf': None, 'files': []}

    png_path = source_dir / f"{stem}.png"
    if png_path.exists():
        result['png'] = png_path
        result['files'].append(png_path)
        result['found'] = True

    pdf_path = source_dir / f"{stem}.pdf"
    if pdf_path.exists():
        result['pdf'] = pdf_path
        result['files'].append(pdf_path)
        result['found'] = True

    return result


def process_figure(figure_entry: Dict, config: Config) -> Dict:
    """Process a single figure based on manifest entry."""
    figure_name = figure_entry['figure']
    subfolder = figure_entry['source_subfolder']
    stem = figure_entry['source_stem']

    result = {
        'figure': figure_name,
        'status': 'pending',
        'source': {'png': None, 'pdf': None},
        'destination': {'png': None, 'pdf': None},
        'issues': [],
        'actions': [],
        'size_info': {}
    }

    log_message("INFO", f"Processing {figure_name}...")

    source_info = locate_source_file(config.BASE_PATH, subfolder, stem)

    if not source_info['found']:
        result['status'] = 'missing'
        result['issues'].append(f"Source file not found in {subfolder}/")
        log_message("ERROR", f"  Source file not found for {figure_name}")
        return result

    for src_path in source_info['files']:
        file_info = get_file_info(src_path)
        result['size_info'][src_path.name] = file_info

        ext = src_path.suffix.lower()
        dest_name = f"{figure_name}{ext}"
        dest_path = config.SUBMISSION_PATH / dest_name

        if ext == '.pdf':
            # PDFs are copied through unchanged -- no pixel requirement
            # applies to them, and PIL cannot open PDFs at all
            shutil.copy2(src_path, dest_path)
            result['destination']['pdf'] = str(dest_path)
            result['source']['pdf'] = str(src_path)
            result['actions'].append(f"Copied {src_path.name} (PDF, no processing needed)")
            log_message("INFO", f"  Copied {src_path.name} (PDF)")
            continue

        # PNG path: validate against the real JASSS width requirement
        is_valid, issues = validate_against_jasss(file_info, config)
        if issues:
            result['issues'].extend(issues)
            for issue in issues:
                log_message("WARNING", f"  {issue}")

        if not is_valid:
            log_message("INFO", f"  Width {file_info.get('width')}px exceeds limit, resizing")
            success = resize_to_max_width(src_path, dest_path, config.MAX_WIDTH_PX)
            if success:
                with Image.open(dest_path) as resized_img:
                    result['actions'].append(
                        f"Resized from {file_info.get('width')}px to "
                        f"{resized_img.width}px wide"
                    )
                result['destination']['png'] = str(dest_path)
                log_message("INFO", f"  Resized successfully")
            else:
                result['issues'].append("Could not resize file")
                log_message("ERROR", f"  Could not resize {figure_name}")
        else:
            shutil.copy2(src_path, dest_path)
            result['destination']['png'] = str(dest_path)
            result['actions'].append(
                f"Copied {src_path.name} ({file_info.get('width')}x{file_info.get('height')}px, within limits)"
            )
            log_message("INFO", f"  Copied {src_path.name} (already compliant)")

        result['source']['png'] = str(src_path)

    result['status'] = 'processed'
    log_message("INFO", f"  Completed processing {figure_name}")
    return result


def generate_report(results: List[Dict], config: Config, summary: Dict):
    """Generate comprehensive validation report."""
    report_path = config.SUBMISSION_PATH / "validation_report.txt"
    log_path = config.SUBMISSION_PATH / "processing_log.json"

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("JASSS FIGURE VALIDATION REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append("SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total Figures: {summary['total']}")
    report_lines.append(f"Processed: {summary['processed']}")
    report_lines.append(f"Resized: {summary['resized']}")
    report_lines.append(f"Failed: {summary['failed']}")
    report_lines.append(f"Missing: {summary['missing']}")
    report_lines.append("")
    report_lines.append("Requirements (per JASSS's actual submission instructions):")
    report_lines.append(f"  Max width: {config.MAX_WIDTH_PX}px (hard requirement)")
    report_lines.append(
        f"  Height guidance: {config.SOFT_MIN_HEIGHT_PX}-{config.SOFT_MAX_HEIGHT_PX}px "
        f"(soft; informational only)"
    )
    report_lines.append("  No file-size or DPI requirement specified by JASSS")
    report_lines.append("")

    report_lines.append("-" * 80)
    report_lines.append("DETAILED RESULTS")
    report_lines.append("-" * 80)
    report_lines.append("")

    for result in results:
        report_lines.append(f"Figure: {result['figure']}")
        report_lines.append(f"  Status: {result['status']}")

        if result['source']['png'] or result['source']['pdf']:
            report_lines.append("  Source:")
            if result['source']['png']:
                report_lines.append(f"    PNG: {result['source']['png']}")
            if result['source']['pdf']:
                report_lines.append(f"    PDF: {result['source']['pdf']}")

        if result['destination']['png'] or result['destination']['pdf']:
            report_lines.append("  Destination:")
            if result['destination']['png']:
                report_lines.append(f"    PNG: {result['destination']['png']}")
            if result['destination']['pdf']:
                report_lines.append(f"    PDF: {result['destination']['pdf']}")

        if result['issues']:
            report_lines.append("  Issues:")
            for issue in result['issues']:
                report_lines.append(f"    - {issue}")

        if result['actions']:
            report_lines.append("  Actions:")
            for action in result['actions']:
                report_lines.append(f"    - {action}")

        report_lines.append("")

    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    with open(log_path, 'w') as f:
        json.dump({
            'summary': summary,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'max_width_px': config.MAX_WIDTH_PX,
                'soft_min_height_px': config.SOFT_MIN_HEIGHT_PX,
                'soft_max_height_px': config.SOFT_MAX_HEIGHT_PX,
                'preferred_formats': config.PREFERRED_FORMATS
            }
        }, f, indent=2, default=str)

    log_message("INFO", f"Report saved: {report_path}")
    log_message("INFO", f"Log saved: {log_path}")


def backup_existing_files(config: Config) -> Path:
    """Backup current submission folder."""
    backup_dest = config.BACKUP_PATH / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dest.mkdir(parents=True, exist_ok=True)

    files_backed_up = 0
    for file in config.SUBMISSION_PATH.glob("Figure_*.*"):
        if file.parent != backup_dest:
            shutil.copy2(file, backup_dest / file.name)
            files_backed_up += 1

    log_message("INFO", f"Backed up {files_backed_up} files to {backup_dest}")
    return backup_dest


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("JASSS FIGURE PREPARATION SCRIPT")
    print("=" * 80)

    config = Config()

    log_message("INFO", f"Base path: {config.BASE_PATH}")
    log_message("INFO", f"Submission path: {config.SUBMISSION_PATH}")
    log_message("INFO", f"Manifest path: {config.MANIFEST_PATH}")
    log_message("INFO", f"Max width: {config.MAX_WIDTH_PX}px")

    config.SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)
    config.BACKUP_PATH.mkdir(parents=True, exist_ok=True)

    backup_path = backup_existing_files(config)

    manifest = load_manifest(config.MANIFEST_PATH)
    if not manifest:
        log_message("ERROR", "No figures found in manifest. Exiting.")
        sys.exit(1)

    log_message("INFO", f"Processing {len(manifest)} figures...")
    print("-" * 80)

    results = []
    summary = {
        'total': len(manifest),
        'processed': 0,
        'resized': 0,
        'failed': 0,
        'missing': 0,
        'timestamp': datetime.now().isoformat()
    }

    for figure_entry in manifest:
        result = process_figure(figure_entry, config)
        results.append(result)

        if result['status'] == 'missing':
            summary['missing'] += 1
        elif result['status'] == 'processed':
            summary['processed'] += 1
            for action in result['actions']:
                if 'resized' in action.lower():
                    summary['resized'] += 1
        else:
            summary['failed'] += 1

    generate_report(results, config, summary)

    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successfully processed: {summary['processed']}")
    print(f"Resized (were over 800px wide): {summary['resized']}")
    print(f"Failed: {summary['failed']}")
    print(f"Missing: {summary['missing']}")
    print(f"")
    print(f"Submission folder: {config.SUBMISSION_PATH}")
    print(f"Validation report: {config.SUBMISSION_PATH / 'validation_report.txt'}")
    print(f"Processing log: {config.SUBMISSION_PATH / 'processing_log.json'}")
    print(f"Backup created at: {backup_path}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)