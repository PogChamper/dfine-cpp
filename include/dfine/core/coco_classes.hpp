#pragma once

namespace dfine {

// The 80 COCO detection class names in contiguous-id order (0..79), matching the
// D-FINE decode's class_id. Purely for display — the library never depends on it.
inline const char* coco_class_name(int id) noexcept {
    static const char* kNames[80] = {"person",        "bicycle",      "car",
                                     "motorcycle",    "airplane",     "bus",
                                     "train",         "truck",        "boat",
                                     "traffic light", "fire hydrant", "stop sign",
                                     "parking meter", "bench",        "bird",
                                     "cat",           "dog",          "horse",
                                     "sheep",         "cow",          "elephant",
                                     "bear",          "zebra",        "giraffe",
                                     "backpack",      "umbrella",     "handbag",
                                     "tie",           "suitcase",     "frisbee",
                                     "skis",          "snowboard",    "sports ball",
                                     "kite",          "baseball bat", "baseball glove",
                                     "skateboard",    "surfboard",    "tennis racket",
                                     "bottle",        "wine glass",   "cup",
                                     "fork",          "knife",        "spoon",
                                     "bowl",          "banana",       "apple",
                                     "sandwich",      "orange",       "broccoli",
                                     "carrot",        "hot dog",      "pizza",
                                     "donut",         "cake",         "chair",
                                     "couch",         "potted plant", "bed",
                                     "dining table",  "toilet",       "tv",
                                     "laptop",        "mouse",        "remote",
                                     "keyboard",      "cell phone",   "microwave",
                                     "oven",          "toaster",      "sink",
                                     "refrigerator",  "book",         "clock",
                                     "vase",          "scissors",     "teddy bear",
                                     "hair drier",    "toothbrush"};
    return (id >= 0 && id < 80) ? kNames[id] : "?";
}

}  // namespace dfine
