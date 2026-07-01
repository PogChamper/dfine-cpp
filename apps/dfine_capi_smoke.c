/*
 * dfine_capi_smoke.c — pure-C smoke test for the D-FINE C ABI.
 *
 * Compiled as C (not C++) so it proves two things at once:
 *   1. dfine/c_api.h is valid, self-contained C (no C++ leaks into the header).
 *   2. No exception escapes the extern "C" boundary — every error path returns
 *      NULL / a default and leaves a message in dfine_last_error().
 *
 * With no engine it runs the ABI/error-path self-tests against a fake image.
 * Given --engine it additionally drives create -> detect -> detect_batch ->
 * free -> destroy on a synthetic buffer to exercise the full happy path.
 *
 * usage: dfine_capi_smoke [--engine E.engine] [--meta E.json] [--threshold 0.5] [--graph]
 */

#include "dfine/c_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

/* Report a check; increments the failure counter when cond is false. */
static void check(int cond, const char* what) {
    if (cond) {
        printf("  [ok]   %s\n", what);
    } else {
        printf("  [FAIL] %s\n", what);
        ++g_failures;
    }
}

/* Fill a freshly malloc'd HWC RGB buffer with a flat gray value. */
static uint8_t* make_gray(int w, int h, int value) {
    size_t n = (size_t)w * (size_t)h * 3u;
    uint8_t* buf = (uint8_t*)malloc(n);
    if (buf) memset(buf, value, n);
    return buf;
}

static void run_selftests(void) {
    printf("[selftest] static helpers + error paths\n");

    check(dfine_version() != NULL && dfine_version()[0] != '\0', "dfine_version() non-empty");
    check(strcmp(dfine_class_name(0), "person") == 0, "class_name(0) == person");
    check(strcmp(dfine_class_name(79), "toothbrush") == 0, "class_name(79) == toothbrush");
    check(strcmp(dfine_class_name(-1), "?") == 0, "class_name(-1) == ?");
    check(strcmp(dfine_class_name(80), "?") == 0, "class_name(80) == ? (no bg / oob)");

    /* NULL engine path -> NULL handle + non-empty error, no crash. */
    check(dfine_detector_create(NULL, NULL) == NULL, "create(NULL) -> NULL");
    check(dfine_last_error()[0] != '\0', "  ... sets last_error");

    /* Bogus engine path -> NULL handle + non-empty error, no crash. */
    check(dfine_detector_create("/nonexistent/does_not_exist.engine", NULL) == NULL,
          "create(bad path) -> NULL");
    check(dfine_last_error()[0] != '\0', "  ... sets last_error");

    /* Introspection on NULL returns safe defaults. */
    check(strcmp(dfine_detector_variant(NULL), "") == 0, "variant(NULL) == \"\"");
    check(dfine_detector_input_width(NULL) == 0, "input_width(NULL) == 0");
    check(dfine_detector_num_classes(NULL) == 0, "num_classes(NULL) == 0");
    check(dfine_detector_max_batch(NULL) == 0, "max_batch(NULL) == 0");

    /* detect on NULL detector -> NULL + error. */
    check(dfine_detector_detect(NULL, NULL, 0, 0, 0, 3, 0, 0.5f) == NULL, "detect(NULL) -> NULL");
    check(dfine_detector_detect_batch(NULL, NULL, 0, 0.5f) == NULL, "detect_batch(NULL) -> NULL");

    /* free / destroy on NULL are no-ops. */
    dfine_detections_free(NULL);
    dfine_detections_free_batch(NULL, 0);
    dfine_detector_destroy(NULL);
    check(1, "free/destroy(NULL) are safe no-ops");
}

static void run_engine_path(const char* engine, const char* meta, float threshold, int use_graph) {
    printf("[engine]   %s\n", engine);

    dfine_options_t opts;
    memset(&opts, 0, sizeof opts);
    opts.threshold = threshold;
    opts.use_cuda_graph = use_graph;

    dfine_detector_t* det = dfine_detector_create_ex(engine, meta, &opts);
    if (!det) {
        printf("  [FAIL] create_ex: %s\n", dfine_last_error());
        ++g_failures;
        return;
    }

    int w = dfine_detector_input_width(det);
    int h = dfine_detector_input_height(det);
    int mb = dfine_detector_max_batch(det);
    printf("  variant=%s input=%dx%d queries=%d classes=%d max_batch=%d\n",
           dfine_detector_variant(det), w, h, dfine_detector_num_queries(det),
           dfine_detector_num_classes(det), mb);
    check(w > 0 && h > 0, "engine reports positive input dims");
    check(dfine_detector_num_classes(det) > 0, "engine reports >0 classes");

    /* Single detect on a synthetic gray image (packed and padded strides). */
    int iw = 640, ih = 480;
    uint8_t* img = make_gray(iw, ih, 128);
    check(img != NULL, "allocate synthetic image");
    if (img) {
        dfine_detections_t* r = dfine_detector_detect(det, img, iw, ih, /*step=*/0, 3, 0, -1.0f);
        check(r != NULL, "detect(packed) -> non-NULL");
        if (r) {
            check(r->count >= 0 && (r->count == 0 || r->detections != NULL),
                  "  ... count/detections consistent");
            printf("  detect: %d detection(s) on gray image\n", r->count);
            dfine_detections_free(r);
        } else {
            printf("  [FAIL] detect: %s\n", dfine_last_error());
        }

        /* Explicit (padded) stride path. */
        dfine_detections_t* r2 = dfine_detector_detect(det, img, iw, ih, iw * 3, 3, 0, 0.5f);
        check(r2 != NULL, "detect(explicit step) -> non-NULL");
        dfine_detections_free(r2);
    }

    /* Batch detect when the engine allows it. */
    if (mb >= 2 && img) {
        dfine_image_t batch[2];
        memset(batch, 0, sizeof batch);
        for (int i = 0; i < 2; ++i) {
            batch[i].data = img;
            batch[i].width = iw;
            batch[i].height = ih;
            batch[i].step = 0;
            batch[i].channels = 3;
            batch[i].is_bgr = 0;
        }
        dfine_detections_t** br = dfine_detector_detect_batch(det, batch, 2, 0.5f);
        check(br != NULL, "detect_batch(2) -> non-NULL");
        if (br) {
            check(br[0] != NULL && br[1] != NULL, "  ... both result sets present");
            printf("  detect_batch: [%d, %d] detections\n", br[0]->count, br[1]->count);
            dfine_detections_free_batch(br, 2);
        } else {
            printf("  [FAIL] detect_batch: %s\n", dfine_last_error());
        }
    }

    free(img);
    dfine_detector_destroy(det);
}

int main(int argc, char** argv) {
    const char* engine = NULL;
    const char* meta = NULL;
    float threshold = 0.5f;
    int use_graph = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("usage: %s [--engine E.engine] [--meta E.json] [--threshold 0.5] [--graph]\n",
                   argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--engine") == 0 && i + 1 < argc) {
            engine = argv[++i];
        } else if (strcmp(argv[i], "--meta") == 0 && i + 1 < argc) {
            meta = argv[++i];
        } else if (strcmp(argv[i], "--threshold") == 0 && i + 1 < argc) {
            threshold = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--graph") == 0) {
            use_graph = 1;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            return 2;
        }
    }

    printf("dfine C-ABI smoke — v%s\n", dfine_version());
    run_selftests();
    if (engine) run_engine_path(engine, meta, threshold, use_graph);

    printf("\n%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures,
           g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
