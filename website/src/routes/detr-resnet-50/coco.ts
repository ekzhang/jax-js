/** Generate a consistent color for a string using a simple hash. */
export function stringToColor(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  // Use HSL for vibrant colors (high saturation, medium lightness)
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 80%, 50%)`;
}

// COCO class names for DETR (indices 0-91, with gaps marked as N/A)
// DETR uses original COCO category IDs which have gaps for unused classes.
// Index 91 is the "no object" class.
// See: https://github.com/facebookresearch/detr/issues/108
export const COCO_CLASSES = [
  "N/A", // 0 - unused
  "person", // 1
  "bicycle", // 2
  "car", // 3
  "motorcycle", // 4
  "airplane", // 5
  "bus", // 6
  "train", // 7
  "truck", // 8
  "boat", // 9
  "traffic light", // 10
  "fire hydrant", // 11
  "N/A", // 12 - unused
  "stop sign", // 13
  "parking meter", // 14
  "bench", // 15
  "bird", // 16
  "cat", // 17
  "dog", // 18
  "horse", // 19
  "sheep", // 20
  "cow", // 21
  "elephant", // 22
  "bear", // 23
  "zebra", // 24
  "giraffe", // 25
  "N/A", // 26 - unused
  "backpack", // 27
  "umbrella", // 28
  "N/A", // 29 - unused
  "N/A", // 30 - unused
  "handbag", // 31
  "tie", // 32
  "suitcase", // 33
  "frisbee", // 34
  "skis", // 35
  "snowboard", // 36
  "sports ball", // 37
  "kite", // 38
  "baseball bat", // 39
  "baseball glove", // 40
  "skateboard", // 41
  "surfboard", // 42
  "tennis racket", // 43
  "bottle", // 44
  "N/A", // 45 - unused
  "wine glass", // 46
  "cup", // 47
  "fork", // 48
  "knife", // 49
  "spoon", // 50
  "bowl", // 51
  "banana", // 52
  "apple", // 53
  "sandwich", // 54
  "orange", // 55
  "broccoli", // 56
  "carrot", // 57
  "hot dog", // 58
  "pizza", // 59
  "donut", // 60
  "cake", // 61
  "chair", // 62
  "couch", // 63
  "potted plant", // 64
  "bed", // 65
  "N/A", // 66 - unused
  "dining table", // 67
  "N/A", // 68 - unused
  "N/A", // 69 - unused
  "toilet", // 70
  "N/A", // 71 - unused
  "tv", // 72
  "laptop", // 73
  "mouse", // 74
  "remote", // 75
  "keyboard", // 76
  "cell phone", // 77
  "microwave", // 78
  "oven", // 79
  "toaster", // 80
  "sink", // 81
  "refrigerator", // 82
  "N/A", // 83 - unused
  "book", // 84
  "clock", // 85
  "vase", // 86
  "scissors", // 87
  "teddy bear", // 88
  "hair drier", // 89
  "toothbrush", // 90
];
