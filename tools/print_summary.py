"""Print the headline numbers from a benchmark JSON (any Python 3)."""
import json
import sys

d = json.load(open(sys.argv[1]))
u = d['ultralytics']
g = u['ground_truth']['all']
print('Ultralytics (FYP adapter): found %d/%d, FP %d/%d, GT IoU median %.4f, centroid median %.3f px, '
      'latency median %.1f p95 %.1f ms' % (g['found'], g['positive_images'], g['false_positives'], g['empty_images'],
                                           g['mask_iou']['median'], g['centroid_px']['median'],
                                           u['latency_ms']['median'], u['latency_ms']['p95']))
for engine, r in d['opencv5'].items():
    a, g, lat = r['agreement'], r['ground_truth']['all'], r['latency_ms']
    print(f'OpenCV 5 [{engine}]: detection agreement {a["detection_decision_agreement"]}/{a["images"]} '
          f'(only cv5 {a["only_opencv5"]}, only ultralytics {a["only_ultralytics"]})')
    print('  vs Ultralytics: IoU min %.5f p05 %.5f median %.5f; centroid median %.4f p95 %.4f max %.4f px; '
          'pixel disagreement median %g max %g; |conf diff| median %.2e max %.2e' % (
              a['mask_iou_vs_ultralytics']['min'], a['mask_iou_vs_ultralytics']['p05'],
              a['mask_iou_vs_ultralytics']['median'], a['centroid_px_vs_ultralytics']['median'],
              a['centroid_px_vs_ultralytics']['p95'], a['centroid_px_vs_ultralytics']['max'],
              a['pixel_disagreement']['median'], a['pixel_disagreement']['max'],
              a['confidence_abs_diff']['median'], a['confidence_abs_diff']['max']))
    print('  vs ground truth: found %d/%d, FP %d, IoU median %.4f, centroid median %.3f px' % (
        g['found'], g['positive_images'], g['false_positives'], g['mask_iou']['median'], g['centroid_px']['median']))
    print('  latency ms: total median %.1f p95 %.1f; forward median %.1f; postprocess median %.1f' % (
        lat['total']['median'], lat['total']['p95'], lat['forward']['median'], lat['postprocess']['median']))
print('env', d['environment'], 'torch threads', d['reference']['torch_threads'])
