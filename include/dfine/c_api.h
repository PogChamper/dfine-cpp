/*
 * dfine/c_api.h — stable C ABI for D-FINE-cpp.
 *
 * Opaque handles, raw-byte image input, heap-allocated result sets. No C++,
 * CUDA, TensorRT, or OpenCV types cross this boundary — safe to #include from
 * pure C, Python ctypes, Rust FFI, Go cgo, or any language with a C FFI layer.
 *
 * Build the library with -DDFINE_BUILD_C_API=ON to enable (the C++ API stays
 * available regardless).
 *
 * Thread safety: a detector handle is NOT thread-safe. Create one per thread.
 * dfine_last_error() is thread-local; dfine_set_log_callback() is process-wide.
 *
 * Quick example (detection):
 *
 *   dfine_detector_t* det = dfine_detector_create("model.engine", NULL);
 *   if (!det) { fprintf(stderr, "%s\n", dfine_last_error()); return 1; }
 *
 *   dfine_detections_t* res = dfine_detector_detect(
 *       det, rgb_data, width, height, width * 3, 3, 0 // is_bgr=0
 *       , 0.5f);
 *   if (!res) { fprintf(stderr, "%s\n", dfine_last_error()); return 1; }
 *   for (int i = 0; i < res->count; ++i) {
 *       const dfine_detection_t* d = &res->detections[i];
 *       printf("%s %.2f  [%.1f %.1f %.1f %.1f]\n",
 *              dfine_class_name(d->class_id), d->score,
 *              d->box.x1, d->box.y1, d->box.x2, d->box.y2);
 *   }
 *   dfine_detections_free(res);
 *   dfine_detector_destroy(det);
 *
 * Design note vs. rf-detr-cpp (the header this is modeled on):
 *   - class_id is the DENSE COCO-80 index 0..79. D-FINE has NO background slot,
 *     so — unlike rf-detr — you do NOT add 1 to get a COCO id; dfine_class_name()
 *     maps 0..79 directly to the COCO class names.
 *   - The core is OpenCV-free and channel-order-agnostic, so the detect entry
 *     point takes explicit `channels` and `is_bgr` arguments (RGB or BGR).
 *   - The log callback carries a severity level (the C++ logger is severity-aware).
 */

#ifndef DFINE_C_API_H
#define DFINE_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * Visibility / export macros
 * ---------------------------------------------------------------------- */
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef DFINE_BUILDING_LIB
#define DFINE_API __declspec(dllexport)
#else
#define DFINE_API __declspec(dllimport)
#endif
#else
#define DFINE_API __attribute__((visibility("default")))
#endif

/* -------------------------------------------------------------------------
 * Opaque handle types
 * ---------------------------------------------------------------------- */
typedef struct dfine_detector_s dfine_detector_t; /* detection engine handle */

/* -------------------------------------------------------------------------
 * Data types
 * ---------------------------------------------------------------------- */

/* Bounding box in original-image pixel coordinates, xyxy format. */
typedef struct dfine_box_s {
    float x1, y1, x2, y2;
} dfine_box_t;

/*
 * Single detection result.
 *
 * class_id : DENSE COCO-80 index (0 = "person", ... 79 = "toothbrush").
 *            D-FINE has no background slot — pass it straight to
 *            dfine_class_name(). (For a fine-tuned model with a different label
 *            set, class_id is the model's own contiguous index and
 *            dfine_class_name — which is COCO-only — will not apply.)
 * score    : sigmoid-activated confidence in [0,1].
 */
typedef struct dfine_detection_s {
    dfine_box_t box;
    int class_id;
    float score;
} dfine_detection_t;

/*
 * Heap-allocated set of detection results. Free with dfine_detections_free().
 * When count == 0, `detections` is NULL — always check `count` before indexing.
 */
typedef struct dfine_detections_s {
    dfine_detection_t* detections;
    int count;
} dfine_detections_t;

/*
 * One image for batch detection (see dfine_detector_detect_batch).
 * Same field meaning as the scalar dfine_detector_detect() arguments.
 */
typedef struct dfine_image_s {
    const uint8_t* data; /* first byte of an HWC uint8 image           */
    int width;           /* pixels                                     */
    int height;          /* pixels                                     */
    int step;            /* row stride in bytes; <=0 => width*channels */
    int channels;        /* must be 3                                  */
    int is_bgr;          /* 0 = RGB, non-zero = BGR                    */
} dfine_image_t;

/*
 * Construction options (dfine_detector_create_ex). Zero-initialize and set only
 * what you need; a NULL pointer means "all defaults".
 */
typedef struct dfine_options_s {
    float threshold;        /* default score threshold; <=0 keeps 0.5        */
    int use_cuda_graph;     /* 0/1: opt-in CUDA-graph replay of the engine.  */
                            /*   Cuts batch-1 launch overhead on single-     */
                            /*   stream (--max-aux-streams 0) engines; a safe */
                            /*   no-op (falls back to enqueueV3) otherwise.   */
    int graph_warmup_iters; /* enqueue cycles before capture (<=0 => 3)      */
} dfine_options_t;

/* -------------------------------------------------------------------------
 * Error reporting
 * ---------------------------------------------------------------------- */

/*
 * Human-readable description of the last error on the calling thread. Valid
 * until the next dfine_* call on the same thread. Never returns NULL (returns
 * "" when no error has occurred).
 */
DFINE_API const char* dfine_last_error(void);

/* -------------------------------------------------------------------------
 * Log callback
 * ---------------------------------------------------------------------- */

/*
 * severity mirrors the C++ dfine::LogSeverity / TensorRT ILogger::Severity:
 *   0 = FATAL, 1 = ERROR, 2 = WARN, 3 = INFO, 4 = VERBOSE.
 */
typedef void (*dfine_log_fn_t)(int severity, const char* message);

/*
 * Override the default stderr logger. Pass NULL to restore the default. The
 * callback receives a null-terminated UTF-8 string and may be called from any
 * thread (including TensorRT's internal threads).
 */
DFINE_API void dfine_set_log_callback(dfine_log_fn_t callback);

/* -------------------------------------------------------------------------
 * Detector lifecycle — dfine_detector_t
 * ---------------------------------------------------------------------- */

/*
 * Create a detector from a TensorRT engine file.
 *
 * engine_path : path to the .engine file (UTF-8, null-terminated).
 * meta_path   : path to the .json sidecar, or NULL to use "<engine_path>.json"
 *               automatically (the runtime also trusts engine bindings over the
 *               sidecar for shape facts, so a missing sidecar is non-fatal).
 *
 * Returns a non-NULL handle on success, NULL on failure (see dfine_last_error).
 */
DFINE_API dfine_detector_t* dfine_detector_create(const char* engine_path, const char* meta_path);

/*
 * Like dfine_detector_create() but with construction options (threshold,
 * CUDA-graph). `opts` may be NULL (equivalent to dfine_detector_create()).
 */
DFINE_API dfine_detector_t* dfine_detector_create_ex(const char* engine_path, const char* meta_path,
                                                     const dfine_options_t* opts);

/* Destroy a detector and free all associated resources. Safe with NULL. */
DFINE_API void dfine_detector_destroy(dfine_detector_t* det);

/* -------------------------------------------------------------------------
 * Introspection (all return safe defaults — "" or 0 — when det is NULL)
 * ---------------------------------------------------------------------- */

/* The returned string is owned by the detector and valid until it is destroyed;
 * copy it if you need to outlive the handle. */
DFINE_API const char* dfine_detector_variant(const dfine_detector_t* det);
DFINE_API int dfine_detector_input_width(const dfine_detector_t* det);
DFINE_API int dfine_detector_input_height(const dfine_detector_t* det);
DFINE_API int dfine_detector_num_queries(const dfine_detector_t* det);
DFINE_API int dfine_detector_num_classes(const dfine_detector_t* det);
/* 1 for a static engine; the profile max for a dynamic engine; 0 if unknown. */
DFINE_API int dfine_detector_max_batch(const dfine_detector_t* det);

/* -------------------------------------------------------------------------
 * Inference
 * ---------------------------------------------------------------------- */

/*
 * Run detection on a single image.
 *
 * data      : first byte of an HWC uint8 image (`channels` bytes per pixel).
 * width     : image width in pixels.
 * height    : image height in pixels.
 * step      : row stride in bytes; pass <=0 for tightly packed (width*channels).
 * channels  : must be 3.
 * is_bgr    : 0 for RGB channel order, non-zero for BGR (e.g. straight from OpenCV).
 * threshold : score threshold in [0,1]; pass < 0 to use the detector's default.
 *
 * Returns a heap-allocated dfine_detections_t* on success (may have count == 0),
 * or NULL on failure. Free the result with dfine_detections_free().
 */
DFINE_API dfine_detections_t* dfine_detector_detect(dfine_detector_t* det, const uint8_t* data,
                                                    int width, int height, int step, int channels,
                                                    int is_bgr, float threshold);

/*
 * Batch detection. Requires an engine built with max_batch >= count.
 *
 * Returns a heap-allocated array of `count` result-set pointers (results[i]
 * corresponds to images[i] and is never NULL within the array), or NULL on
 * failure. Free the whole thing with dfine_detections_free_batch().
 */
DFINE_API dfine_detections_t** dfine_detector_detect_batch(dfine_detector_t* det,
                                                           const dfine_image_t* images, int count,
                                                           float threshold);

/* -------------------------------------------------------------------------
 * Result memory management
 * ---------------------------------------------------------------------- */

/* Free a result set from dfine_detector_detect(). Safe with NULL. */
DFINE_API void dfine_detections_free(dfine_detections_t* dets);

/* Free a batch result array from dfine_detector_detect_batch(). Safe with NULL. */
DFINE_API void dfine_detections_free_batch(dfine_detections_t** results, int count);

/* -------------------------------------------------------------------------
 * Static helpers
 * ---------------------------------------------------------------------- */

/*
 * Name of a COCO-80 class by dense id (0..79), e.g. dfine_class_name(0)=="person".
 * Returns "?" out of range. Display helper only — the library never depends on it,
 * and it does NOT apply to models fine-tuned on a non-COCO label set.
 */
DFINE_API const char* dfine_class_name(int class_id);

/*
 * Model-aware class name: the engine sidecar's class_names entry when present
 * (custom label sets), else the COCO-80 table for 80-class engines, else
 * "class_<id>". Returns "" for a NULL detector or an id outside
 * 0..num_classes-1. The pointer stays valid until dfine_detector_destroy().
 */
DFINE_API const char* dfine_detector_class_name(const dfine_detector_t* det, int class_id);

/* Library version string, e.g. "0.1.0". */
DFINE_API const char* dfine_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DFINE_C_API_H */
