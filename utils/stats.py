"""
Statistics Utilities
====================

Functions for calculating detection statistics and analysis.
"""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


def calculate_detection_stats(
    results: List[Dict[str, Any]],
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Calculate statistics from detection results.

    Args:
        results: List of detection result dictionaries with keys:
                 'filename', 'num_detections', 'confidences', 'class_ids'
        class_names: List of class names (optional)

    Returns:
        Dictionary containing detection statistics
    """
    total_images = len(results)
    total_detections = sum(r['num_detections'] for r in results)
    images_with_detections = sum(1 for r in results if r['num_detections'] > 0)

    # Calculate per-class statistics
    class_counts = defaultdict(int)
    all_confidences = []

    for result in results:
        confidences = result.get('confidences', [])
        class_ids = result.get('class_ids', [])

        all_confidences.extend(confidences)

        for cls_id in class_ids:
            cls_id = int(cls_id)
            if class_names and cls_id < len(class_names):
                class_counts[class_names[cls_id]] += 1
            else:
                class_counts[f'Class {cls_id}'] += 1

    # Calculate confidence statistics
    if all_confidences:
        conf_array = np.array(all_confidences)
        confidence_stats = {
            'mean': float(conf_array.mean()),
            'std': float(conf_array.std()),
            'min': float(conf_array.min()),
            'max': float(conf_array.max()),
            'median': float(np.median(conf_array))
        }
    else:
        confidence_stats = {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }

    return {
        'total_images': total_images,
        'total_detections': total_detections,
        'images_with_detections': images_with_detections,
        'avg_detections_per_image': total_detections / total_images if total_images > 0 else 0,
        'class_counts': dict(class_counts),
        'confidence_stats': confidence_stats,
        'per_image_results': results
    }


def print_detection_summary(stats: Dict[str, Any]) -> None:
    """
    Print a formatted summary of detection statistics.

    Args:
        stats: Statistics dictionary from calculate_detection_stats
    """
    print("\n" + "="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with detections: {stats['images_with_detections']}")
    print(f"Total detections:       {stats['total_detections']}")
    print(f"Avg detections/image:   {stats['avg_detections_per_image']:.2f}")

    # Print confidence statistics
    conf_stats = stats['confidence_stats']
    print(f"\nConfidence Statistics:")
    print(f"  Mean:   {conf_stats['mean']:.3f}")
    print(f"  Std:    {conf_stats['std']:.3f}")
    print(f"  Median: {conf_stats['median']:.3f}")
    print(f"  Range:  [{conf_stats['min']:.3f}, {conf_stats['max']:.3f}]")

    # Print class distribution
    class_counts = stats['class_counts']
    if class_counts:
        print(f"\nClass Distribution:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_detections']
                          * 100) if stats['total_detections'] > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")

    print("="*80)
