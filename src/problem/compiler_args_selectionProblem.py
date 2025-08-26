import random
import os
import shutil
import time
from pathlib import Path
from os import path
import ck.kernel as ck

from src.problem.base_problem import BaseProblem
from src.types_ import *

all_bench_program_list: List[Dict[str, Union[int, str]]] = [
    {'id': 1, 'dataset': 'cbench', 'program': 'cbench-automotive-bitcount'},
    {'id': 2, 'dataset': 'cbench', 'program': 'cbench-automotive-qsort1'},
    {'id': 3, 'dataset': 'cbench', 'program': 'cbench-automotive-susan'},
    {'id': 4, 'dataset': 'cbench', 'program': 'cbench-bzip2'},
    {'id': 5, 'dataset': 'cbench', 'program': 'cbench-consumer-jpeg-c'},
    {'id': 6, 'dataset': 'cbench', 'program': 'cbench-consumer-jpeg-d'},
    {'id': 7, 'dataset': 'cbench', 'program': 'cbench-consumer-lame'},
    {'id': 8, 'dataset': 'cbench', 'program': 'cbench-consumer-tiff2bw'},
    {'id': 9, 'dataset': 'cbench', 'program': 'cbench-consumer-tiff2dither'},
    {'id': 10, 'dataset': 'cbench', 'program': 'cbench-consumer-tiff2median'},
    {'id': 11, 'dataset': 'cbench', 'program': 'cbench-consumer-tiff2rgba'},
    {'id': 12, 'dataset': 'cbench', 'program': 'cbench-network-dijkstra'},
    {'id': 13, 'dataset': 'cbench', 'program': 'cbench-network-patricia'},
    {'id': 14, 'dataset': 'cbench', 'program': 'cbench-office-stringsearch2'},
    {'id': 15, 'dataset': 'cbench', 'program': 'cbench-security-blowfish'},
    {'id': 16, 'dataset': 'cbench', 'program': 'cbench-security-pgp'},
    {'id': 17, 'dataset': 'cbench', 'program': 'cbench-security-rijndael'},
    {'id': 18, 'dataset': 'cbench', 'program': 'cbench-security-sha'},
    {'id': 19, 'dataset': 'cbench', 'program': 'cbench-telecom-adpcm-c'},
    {'id': 20, 'dataset': 'cbench', 'program': 'cbench-telecom-adpcm-d'},
    {'id': 21, 'dataset': 'cbench', 'program': 'cbench-telecom-crc32'},
    {'id': 22, 'dataset': 'cbench', 'program': 'cbench-telecom-gsm'},
    {'id': 23, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-2mm'},
    {'id': 24, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-3mm'},
    {'id': 25, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-adi'},
    {'id': 26, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-atax'},
    {'id': 27, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-bicg'},
    {'id': 28, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-cholesky'},
    {'id': 29, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-doitgen'},
    {'id': 30, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-durbin'},
    {'id': 31, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-dynprog'},
    {'id': 32, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-fdtd-2d'},
    {'id': 33, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-fdtd-apml'},
    {'id': 34, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-gemm'},
    {'id': 35, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-gemver'},
    {'id': 36, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-gesummv'},
    {'id': 37, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-gramschmidt'},
    {'id': 38, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-jacobi-1d-imper'},
    {'id': 39, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-jacobi-2d-imper'},
    {'id': 40, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-lu'},
    {'id': 41, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-ludcmp'},
    {'id': 42, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-medley-floyd-warshall'},
    {'id': 43, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-medley-reg-detect'},
    {'id': 44, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-mvt'},
    {'id': 45, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-seidel-2d'},
    {'id': 46, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-symm'},
    {'id': 47, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-syr2k'},
    {'id': 48, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-syrk'},
    {'id': 49, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-trisolv'},
    {'id': 50, 'dataset': 'polybench-cpu', 'program': 'polybench-cpu-trmm'}
]

# Some Program that can compile fast, select a subset to save time
_candidate_program_index = [8, 18, 23, 31, 46, 40, 25, 4, 20, 10, 16, 2, 13, 17, 48, 42, 47, 43, 41, 26, 27, 11,
                            49, 7, 30]

_tot_opt_list = [{"compile_flag": ["-fauto-inc-dec", "-fno-auto-inc-dec"], "flag_id": 1, "conflict_list": []},
                 {"compile_flag": ["-fbranch-count-reg", "-fno-branch-count-reg"], "flag_id": 2, "conflict_list": []},
                 {"compile_flag": ["-fbranch-probabilities", "-fno-branch-probabilities"], "flag_id": 3,
                  "conflict_list": []},
                 {"compile_flag": ["-fbranch-target-load-optimize", "-fno-branch-target-load-optimize"], "flag_id": 4,
                  "conflict_list": []},
                 {"compile_flag": ["-fbranch-target-load-optimize2", "-fno-branch-target-load-optimize2"], "flag_id": 5,
                  "conflict_list": []},
                 {"compile_flag": ["-fbtr-bb-exclusive", "-fno-btr-bb-exclusive"], "flag_id": 6, "conflict_list": []},
                 {"compile_flag": ["-fcaller-saves", "-fno-caller-saves"], "flag_id": 7, "conflict_list": []},
                 {"compile_flag": ["-fcheck-data-deps", "-fno-check-data-deps"], "flag_id": 8, "conflict_list": []},
                 {"compile_flag": ["-fcombine-stack-adjustments", "-fno-combine-stack-adjustments"], "flag_id": 9,
                  "conflict_list": []},
                 {"compile_flag": ["-fcompare-elim", "-fno-compare-elim"], "flag_id": 10, "conflict_list": []},
                 {"compile_flag": ["-fconserve-stack", "-fno-conserve-stack"], "flag_id": 11, "conflict_list": []},
                 {"compile_flag": ["-fcprop-registers", "-fno-cprop-registers"], "flag_id": 12, "conflict_list": []},
                 {"compile_flag": ["-fcrossjumping", "-fno-crossjumping"], "flag_id": 13, "conflict_list": []},
                 {"compile_flag": ["-fcse-follow-jumps", "-fno-cse-follow-jumps"], "flag_id": 14, "conflict_list": []},
                 {"compile_flag": ["-fcse-skip-blocks", "-fno-cse-skip-blocks"], "flag_id": 15, "conflict_list": []},
                 {"compile_flag": ["-fcx-fortran-rules", "-fno-cx-fortran-rules"], "flag_id": 16, "conflict_list": []},
                 {"compile_flag": ["-fcx-limited-range", "-fno-cx-limited-range"], "flag_id": 17, "conflict_list": []},
                 {"compile_flag": ["-fdata-sections", "-fno-data-sections"], "flag_id": 18, "conflict_list": []},
                 {"compile_flag": ["-fdce", "-fno-dce"], "flag_id": 19, "conflict_list": []},
                 {"compile_flag": ["-fdefer-pop", "-fno-defer-pop"], "flag_id": 20, "conflict_list": []},
                 {"compile_flag": ["-fdelete-null-pointer-checks", "-fno-delete-null-pointer-checks"], "flag_id": 21,
                  "conflict_list": []},
                 {"compile_flag": ["-fdevirtualize", "-fno-devirtualize"], "flag_id": 22, "conflict_list": []},
                 {"compile_flag": ["-fdevirtualize-speculatively", "-fno-devirtualize-speculatively"], "flag_id": 23,
                  "conflict_list": []}, {"compile_flag": ["-fdse", "-fno-dse"], "flag_id": 24, "conflict_list": []},
                 {"compile_flag": ["-fearly-inlining", "-fno-early-inlining"], "flag_id": 25, "conflict_list": []},
                 {"compile_flag": ["-fexpensive-optimizations", "-fno-expensive-optimizations"], "flag_id": 26,
                  "conflict_list": []},
                 {"compile_flag": ["-ffast-math", "-fno-fast-math"], "flag_id": 27, "conflict_list": []},
                 {"compile_flag": ["-ffinite-math-only", "-fno-finite-math-only"], "flag_id": 28, "conflict_list": []},
                 {"compile_flag": ["-ffloat-store", "-fno-float-store"], "flag_id": 29, "conflict_list": []},
                 {"compile_flag": ["-fforward-propagate", "-fno-forward-propagate"], "flag_id": 30,
                  "conflict_list": []},
                 {"compile_flag": ["-ffunction-cse", "-fno-function-cse"], "flag_id": 31, "conflict_list": []},
                 {"compile_flag": ["-ffunction-sections", "-fno-function-sections"], "flag_id": 32,
                  "conflict_list": []}, {"compile_flag": ["-fgcse", "-fno-gcse"], "flag_id": 33, "conflict_list": []},
                 {"compile_flag": ["-fgcse-after-reload", "-fno-gcse-after-reload"], "flag_id": 34,
                  "conflict_list": []},
                 {"compile_flag": ["-fgcse-las", "-fno-gcse-las"], "flag_id": 35, "conflict_list": []},
                 {"compile_flag": ["-fgcse-lm", "-fno-gcse-lm"], "flag_id": 36, "conflict_list": []},
                 {"compile_flag": ["-fgcse-sm", "-fno-gcse-sm"], "flag_id": 37, "conflict_list": []},
                 {"compile_flag": ["-fgraphite-identity", "-fno-graphite-identity"], "flag_id": 38,
                  "conflict_list": []},
                 {"compile_flag": ["-fguess-branch-probability", "-fno-guess-branch-probability"], "flag_id": 39,
                  "conflict_list": []},
                 {"compile_flag": ["-fhoist-adjacent-loads", "-fno-hoist-adjacent-loads"], "flag_id": 40,
                  "conflict_list": []},
                 {"compile_flag": ["-fif-conversion", "-fno-if-conversion"], "flag_id": 41, "conflict_list": []},
                 {"compile_flag": ["-fif-conversion2", "-fno-if-conversion2"], "flag_id": 42, "conflict_list": []},
                 {"compile_flag": ["-findirect-inlining", "-fno-indirect-inlining"], "flag_id": 43,
                  "conflict_list": []},
                 {"compile_flag": ["-finline", "-fno-inline"], "flag_id": 44, "conflict_list": []},
                 {"compile_flag": ["-finline-functions", "-fno-inline-functions"], "flag_id": 45, "conflict_list": []},
                 {"compile_flag": ["-finline-functions-called-once", "-fno-inline-functions-called-once"],
                  "flag_id": 46, "conflict_list": []},
                 {"compile_flag": ["-finline-small-functions", "-fno-inline-small-functions"], "flag_id": 47,
                  "conflict_list": []},
                 {"compile_flag": ["-fipa-cp", "-fno-ipa-cp"], "flag_id": 48, "conflict_list": []},
                 {"compile_flag": ["-fipa-cp-clone", "-fno-ipa-cp-clone"], "flag_id": 49, "conflict_list": []},
                 {"compile_flag": ["-fipa-pta", "-fno-ipa-pta"], "flag_id": 50, "conflict_list": []},
                 {"compile_flag": ["-fipa-pure-const", "-fno-ipa-pure-const"], "flag_id": 51, "conflict_list": []},
                 {"compile_flag": ["-fipa-reference", "-fno-ipa-reference"], "flag_id": 52, "conflict_list": []},
                 {"compile_flag": ["-fipa-sra", "-fno-ipa-sra"], "flag_id": 53, "conflict_list": []},
                 {"compile_flag": ["-fira-hoist-pressure", "-fno-ira-hoist-pressure"], "flag_id": 54,
                  "conflict_list": []},
                 {"compile_flag": ["-fira-loop-pressure", "-fno-ira-loop-pressure"], "flag_id": 55,
                  "conflict_list": []},
                 {"compile_flag": ["-fira-share-save-slots", "-fno-ira-share-save-slots"], "flag_id": 56,
                  "conflict_list": []},
                 {"compile_flag": ["-fira-share-spill-slots", "-fno-ira-share-spill-slots"], "flag_id": 57,
                  "conflict_list": []},
                 {"compile_flag": ["-fisolate-erroneous-paths-attribute", "-fno-isolate-erroneous-paths-attribute"],
                  "flag_id": 58, "conflict_list": []},
                 {"compile_flag": ["-fisolate-erroneous-paths-dereference", "-fno-isolate-erroneous-paths-dereference"],
                  "flag_id": 59, "conflict_list": []},
                 {"compile_flag": ["-fivopts", "-fno-ivopts"], "flag_id": 60, "conflict_list": []},
                 {"compile_flag": ["-fkeep-inline-functions", "-fno-keep-inline-functions"], "flag_id": 61,
                  "conflict_list": []},
                 {"compile_flag": ["-fkeep-static-consts", "-fno-keep-static-consts"], "flag_id": 62,
                  "conflict_list": []},
                 {"compile_flag": ["-flive-range-shrinkage", "-fno-live-range-shrinkage"], "flag_id": 63,
                  "conflict_list": []},
                 {"compile_flag": ["-floop-block", "-fno-loop-block"], "flag_id": 64, "conflict_list": []},
                 {"compile_flag": ["-floop-interchange", "-fno-loop-interchange"], "flag_id": 65, "conflict_list": []},
                 {"compile_flag": ["-floop-nest-optimize", "-fno-loop-nest-optimize"], "flag_id": 66,
                  "conflict_list": []},
                 {"compile_flag": ["-floop-parallelize-all", "-fno-loop-parallelize-all"], "flag_id": 67,
                  "conflict_list": []},
                 {"compile_flag": ["-floop-strip-mine", "-fno-loop-strip-mine"], "flag_id": 68, "conflict_list": []},
                 {"compile_flag": ["-fmath-errno", "-fno-math-errno"], "flag_id": 69, "conflict_list": []},
                 {"compile_flag": ["-fmerge-all-constants", "-fno-merge-all-constants"], "flag_id": 70,
                  "conflict_list": []},
                 {"compile_flag": ["-fmerge-constants", "-fno-merge-constants"], "flag_id": 71, "conflict_list": []},
                 {"compile_flag": ["-fmodulo-sched", "-fno-modulo-sched"], "flag_id": 72, "conflict_list": []},
                 {"compile_flag": ["-fmodulo-sched-allow-regmoves", "-fno-modulo-sched-allow-regmoves"], "flag_id": 73,
                  "conflict_list": []},
                 {"compile_flag": ["-fmove-loop-invariants", "-fno-move-loop-invariants"], "flag_id": 74,
                  "conflict_list": []},
                 {"compile_flag": ["-fomit-frame-pointer", "-fno-omit-frame-pointer"], "flag_id": 75,
                  "conflict_list": []},
                 {"compile_flag": ["-foptimize-sibling-calls", "-fno-optimize-sibling-calls"], "flag_id": 76,
                  "conflict_list": []},
                 {"compile_flag": ["-fpartial-inlining", "-fno-partial-inlining"], "flag_id": 77, "conflict_list": []},
                 {"compile_flag": ["-fpeel-loops", "-fno-peel-loops"], "flag_id": 78, "conflict_list": []},
                 {"compile_flag": ["-fpeephole", "-fno-peephole"], "flag_id": 79, "conflict_list": []},
                 {"compile_flag": ["-fpeephole2", "-fno-peephole2"], "flag_id": 80, "conflict_list": []},
                 {"compile_flag": ["-fpredictive-commoning", "-fno-predictive-commoning"], "flag_id": 81,
                  "conflict_list": []},
                 {"compile_flag": ["-freciprocal-math", "-fno-reciprocal-math"], "flag_id": 82, "conflict_list": []},
                 {"compile_flag": ["-free", "-fno-ree"], "flag_id": 83, "conflict_list": []},
                 {"compile_flag": ["-frename-registers", "-fno-rename-registers"], "flag_id": 84, "conflict_list": []},
                 {"compile_flag": ["-freorder-blocks", "-fno-reorder-blocks"], "flag_id": 85, "conflict_list": []},
                 {"compile_flag": ["-freorder-blocks-and-partition", "-fno-reorder-blocks-and-partition"],
                  "flag_id": 86, "conflict_list": []},
                 {"compile_flag": ["-freorder-functions", "-fno-reorder-functions"], "flag_id": 87,
                  "conflict_list": []},
                 {"compile_flag": ["-frerun-cse-after-loop", "-fno-rerun-cse-after-loop"], "flag_id": 88,
                  "conflict_list": []},
                 {"compile_flag": ["-freschedule-modulo-scheduled-loops", "-fno-reschedule-modulo-scheduled-loops"],
                  "flag_id": 89, "conflict_list": []},
                 {"compile_flag": ["-frounding-math", "-fno-rounding-math"], "flag_id": 90, "conflict_list": []},
                 {"compile_flag": ["-fsched-critical-path-heuristic", "-fno-sched-critical-path-heuristic"],
                  "flag_id": 91, "conflict_list": []},
                 {"compile_flag": ["-fsched-dep-count-heuristic", "-fno-sched-dep-count-heuristic"], "flag_id": 92,
                  "conflict_list": []},
                 {"compile_flag": ["-fsched-group-heuristic", "-fno-sched-group-heuristic"], "flag_id": 93,
                  "conflict_list": []},
                 {"compile_flag": ["-fsched-interblock", "-fno-sched-interblock"], "flag_id": 94, "conflict_list": []},
                 {"compile_flag": ["-fsched-last-insn-heuristic", "-fno-sched-last-insn-heuristic"], "flag_id": 95,
                  "conflict_list": []},
                 {"compile_flag": ["-fsched-pressure", "-fno-sched-pressure"], "flag_id": 96, "conflict_list": []},
                 {"compile_flag": ["-fsched-rank-heuristic", "-fno-sched-rank-heuristic"], "flag_id": 97,
                  "conflict_list": []},
                 {"compile_flag": ["-fsched-spec", "-fno-sched-spec"], "flag_id": 98, "conflict_list": []},
                 {"compile_flag": ["-fsched-spec-insn-heuristic", "-fno-sched-spec-insn-heuristic"], "flag_id": 99,
                  "conflict_list": []},
                 {"compile_flag": ["-fsched-spec-load", "-fno-sched-spec-load"], "flag_id": 100, "conflict_list": []},
                 {"compile_flag": ["-fsched-spec-load-dangerous", "-fno-sched-spec-load-dangerous"], "flag_id": 101,
                  "conflict_list": []},
                 {"compile_flag": ["-fsched2-use-superblocks", "-fno-sched2-use-superblocks"], "flag_id": 102,
                  "conflict_list": []},
                 {"compile_flag": ["-fschedule-insns", "-fno-schedule-insns"], "flag_id": 103, "conflict_list": []},
                 {"compile_flag": ["-fschedule-insns2", "-fno-schedule-insns2"], "flag_id": 104, "conflict_list": []},
                 {"compile_flag": ["-fsel-sched-pipelining", "-fno-sel-sched-pipelining"], "flag_id": 105,
                  "conflict_list": []},
                 {"compile_flag": ["-fsel-sched-pipelining-outer-loops", "-fno-sel-sched-pipelining-outer-loops"],
                  "flag_id": 106, "conflict_list": []},
                 {"compile_flag": ["-fselective-scheduling", "-fno-selective-scheduling"], "flag_id": 107,
                  "conflict_list": []},
                 {"compile_flag": ["-fselective-scheduling2", "-fno-selective-scheduling2"], "flag_id": 108,
                  "conflict_list": []},
                 {"compile_flag": ["-fshrink-wrap", "-fno-shrink-wrap"], "flag_id": 109, "conflict_list": []},
                 {"compile_flag": ["-fsignaling-nans", "-fno-signaling-nans"], "flag_id": 110, "conflict_list": []},
                 {"compile_flag": ["-fsigned-zeros", "-fno-signed-zeros"], "flag_id": 111, "conflict_list": []},
                 {"compile_flag": ["-fsingle-precision-constant", "-fno-single-precision-constant"], "flag_id": 112,
                  "conflict_list": []},
                 {"compile_flag": ["-fsplit-ivs-in-unroller", "-fno-split-ivs-in-unroller"], "flag_id": 113,
                  "conflict_list": []},
                 {"compile_flag": ["-fsplit-wide-types", "-fno-split-wide-types"], "flag_id": 114, "conflict_list": []},
                 {"compile_flag": ["-fstrict-aliasing", "-fno-strict-aliasing"], "flag_id": 115, "conflict_list": []},
                 {"compile_flag": ["-fstrict-overflow", "-fno-strict-overflow"], "flag_id": 116, "conflict_list": []},
                 {"compile_flag": ["-fthread-jumps", "-fno-thread-jumps"], "flag_id": 117, "conflict_list": []},
                 {"compile_flag": ["-ftoplevel-reorder", "-fno-toplevel-reorder"], "flag_id": 118, "conflict_list": []},
                 {"compile_flag": ["-ftracer", "-fno-tracer"], "flag_id": 119, "conflict_list": []},
                 {"compile_flag": ["-ftrapping-math", "-fno-trapping-math"], "flag_id": 120, "conflict_list": []},
                 {"compile_flag": ["-ftree-bit-ccp", "-fno-tree-bit-ccp"], "flag_id": 121, "conflict_list": []},
                 {"compile_flag": ["-ftree-builtin-call-dce", "-fno-tree-builtin-call-dce"], "flag_id": 122,
                  "conflict_list": []},
                 {"compile_flag": ["-ftree-ccp", "-fno-tree-ccp"], "flag_id": 123, "conflict_list": []},
                 {"compile_flag": ["-ftree-ch", "-fno-tree-ch"], "flag_id": 124, "conflict_list": []},
                 {"compile_flag": ["-ftree-coalesce-vars", "-fno-tree-coalesce-vars"], "flag_id": 125,
                  "conflict_list": []},
                 {"compile_flag": ["-ftree-copy-prop", "-fno-tree-copy-prop"], "flag_id": 126, "conflict_list": []},
                 {"compile_flag": ["-ftree-copyrename", "-fno-tree-copyrename"], "flag_id": 127, "conflict_list": []},
                 {"compile_flag": ["-ftree-dce", "-fno-tree-dce"], "flag_id": 128, "conflict_list": []},
                 {"compile_flag": ["-ftree-dominator-opts", "-fno-tree-dominator-opts"], "flag_id": 129,
                  "conflict_list": []},
                 {"compile_flag": ["-ftree-dse", "-fno-tree-dse"], "flag_id": 130, "conflict_list": []},
                 {"compile_flag": ["-ftree-forwprop", "-fno-tree-forwprop"], "flag_id": 131, "conflict_list": []},
                 {"compile_flag": ["-ftree-fre", "-fno-tree-fre"], "flag_id": 132, "conflict_list": []},
                 {"compile_flag": ["-ftree-loop-distribute-patterns", "-fno-tree-loop-distribute-patterns"],
                  "flag_id": 133, "conflict_list": []},
                 {"compile_flag": ["-ftree-loop-distribution", "-fno-tree-loop-distribution"], "flag_id": 134,
                  "conflict_list": []},
                 {"compile_flag": ["-ftree-loop-if-convert", "-fno-tree-loop-if-convert"], "flag_id": 135,
                  "conflict_list": []},
                 {"compile_flag": ["-ftree-loop-if-convert-stores", "-fno-tree-loop-if-convert-stores"], "flag_id": 136,
                  "conflict_list": []},
                 {"compile_flag": ["-ftree-loop-im", "-fno-tree-loop-im"], "flag_id": 137, "conflict_list": []},
                 {"compile_flag": ["-ftree-loop-ivcanon", "-fno-tree-loop-ivcanon"], "flag_id": 138,
                  "conflict_list": []},
                 {"compile_flag": ["-ftree-loop-linear", "-fno-tree-loop-linear"], "flag_id": 139, "conflict_list": []},
                 {"compile_flag": ["-ftree-loop-optimize", "-fno-tree-loop-optimize"], "flag_id": 140,
                  "conflict_list": []},
                 {"compile_flag": ["-ftree-loop-vectorize", "-fno-tree-loop-vectorize"], "flag_id": 141,
                  "conflict_list": []},
                 {"compile_flag": ["-ftree-partial-pre", "-fno-tree-partial-pre"], "flag_id": 142, "conflict_list": []},
                 {"compile_flag": ["-ftree-phiprop", "-fno-tree-phiprop"], "flag_id": 143, "conflict_list": []},
                 {"compile_flag": ["-ftree-pre", "-fno-tree-pre"], "flag_id": 144, "conflict_list": []},
                 {"compile_flag": ["-ftree-pta", "-fno-tree-pta"], "flag_id": 145, "conflict_list": []},
                 {"compile_flag": ["-ftree-reassoc", "-fno-tree-reassoc"], "flag_id": 146, "conflict_list": []},
                 {"compile_flag": ["-ftree-sink", "-fno-tree-sink"], "flag_id": 147, "conflict_list": []},
                 {"compile_flag": ["-ftree-slsr", "-fno-tree-slsr"], "flag_id": 148, "conflict_list": []},
                 {"compile_flag": ["-ftree-sra", "-fno-tree-sra"], "flag_id": 149, "conflict_list": []},
                 {"compile_flag": ["-ftree-switch-conversion", "-fno-tree-switch-conversion"], "flag_id": 150,
                  "conflict_list": []},
                 {"compile_flag": ["-ftree-tail-merge", "-fno-tree-tail-merge"], "flag_id": 151, "conflict_list": []},
                 {"compile_flag": ["-ftree-ter", "-fno-tree-ter"], "flag_id": 152, "conflict_list": []},
                 {"compile_flag": ["-ftree-vectorize", "-fno-tree-vectorize"], "flag_id": 153, "conflict_list": []},
                 {"compile_flag": ["-ftree-vrp", "-fno-tree-vrp"], "flag_id": 154, "conflict_list": []},
                 {"compile_flag": ["-funroll-all-loops", "-fno-unroll-all-loops"], "flag_id": 155, "conflict_list": []},
                 {"compile_flag": ["-funroll-loops", "-fno-unroll-loops"], "flag_id": 156, "conflict_list": []},
                 {"compile_flag": ["-funsafe-loop-optimizations", "-fno-unsafe-loop-optimizations"], "flag_id": 157,
                  "conflict_list": []},
                 {"compile_flag": ["-funsafe-math-optimizations", "-fno-unsafe-math-optimizations"], "flag_id": 158,
                  "conflict_list": []},
                 {"compile_flag": ["-funswitch-loops", "-fno-unswitch-loops"], "flag_id": 159, "conflict_list": []},
                 {"compile_flag": ["-fuse-linker-plugin", "-fno-use-linker-plugin"], "flag_id": 160,
                  "conflict_list": []},
                 {"compile_flag": ["-fvariable-expansion-in-unroller", "-fno-variable-expansion-in-unroller"],
                  "flag_id": 161, "conflict_list": []},
                 {"compile_flag": ["-fvect-cost-model", "-fno-vect-cost-model"], "flag_id": 162, "conflict_list": []},
                 {"compile_flag": ["-fvpt", "-fno-vpt"], "flag_id": 163, "conflict_list": []},
                 {"compile_flag": ["-fweb", "-fno-web"], "flag_id": 164, "conflict_list": []},
                 {"compile_flag": ["-fzero-initialized-in-bss", "-fno-zero-initialized-in-bss"], "flag_id": 165,
                  "conflict_list": []},
                 {"compile_flag": ["", "-fexcess-precision=fast"], "flag_id": 166, "conflict_list": []},
                 {"compile_flag": ["", "-fexcess-precision=standard"], "flag_id": 167, "conflict_list": [166]},
                 {"compile_flag": ["", "-ffp-contract=fast"], "flag_id": 168, "conflict_list": []},
                 {"compile_flag": ["", "-ffp-contract=on"], "flag_id": 169, "conflict_list": [168]},
                 {"compile_flag": ["", "-ffp-contract=off"], "flag_id": 170, "conflict_list": [168, 169]},
                 {"compile_flag": ["", "-flifetime-dse=0"], "flag_id": 171, "conflict_list": []},
                 {"compile_flag": ["", "-flifetime-dse=1"], "flag_id": 172, "conflict_list": [171]},
                 {"compile_flag": ["", "-flifetime-dse=2"], "flag_id": 173, "conflict_list": [171, 172]},
                 {"compile_flag": ["", "-fsimd-cost-model=dynamic"], "flag_id": 174, "conflict_list": []},
                 {"compile_flag": ["", "-fsimd-cost-model=cheap"], "flag_id": 175, "conflict_list": [174]},
                 {"compile_flag": ["", "-fsimd-cost-model=unlimited"], "flag_id": 176, "conflict_list": [174, 175]},
                 {"compile_flag": ["", "-fstack-reuse=none"], "flag_id": 177, "conflict_list": []},
                 {"compile_flag": ["", "-fstack-reuse=named_vars"], "flag_id": 178, "conflict_list": [177]},
                 {"compile_flag": ["", "-fstack-reuse=all"], "flag_id": 179, "conflict_list": [177, 178]},
                 {"compile_flag": ["", "-fvect-cost-model=unlimited"], "flag_id": 180, "conflict_list": []},
                 {"compile_flag": ["", "-fvect-cost-model=dynamic"], "flag_id": 181, "conflict_list": [180]},
                 {"compile_flag": ["", "-fvect-cost-model=cheap"], "flag_id": 182, "conflict_list": [180, 181]},
                 {"compile_flag": ["", "-fira-algorithm=CB"], "flag_id": 183, "conflict_list": []},
                 {"compile_flag": ["", "-fira-algorithm=priority"], "flag_id": 184, "conflict_list": [183]},
                 {"compile_flag": ["", "-fira-region=all"], "flag_id": 185, "conflict_list": []},
                 {"compile_flag": ["", "-fira-region=mixed"], "flag_id": 186, "conflict_list": [185]},
                 {"compile_flag": ["", "-fira-region=one"], "flag_id": 187, "conflict_list": [185, 186]}]

proj_tmp_path = Path(path.dirname(path.abspath(__file__)), "../../tmp")


def system_with_timeout_without_output(i):
    import subprocess
    import time

    cmd = i['cmd']

    rc = 0

    to = i.get('timeout', '')

    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if to != '':
        xto = float(to)

        t0 = time.time()
        t = 0
        tx = float(i['timeout'])

        while p.poll() == None and t < xto:
            time.sleep(0.1)
            t = time.time() - t0

        if t >= xto and p.poll() == None:
            ck.system_with_timeout_kill(p)
            return {'return': 8, 'error': 'process timed out and had been terminated'}
    else:
        p.wait()

    rc = p.returncode
    return {'return': 0, 'return_code': rc}


# Overwrite the function to shield the output in the compile process
ck.system_with_timeout = system_with_timeout_without_output


class CompilerArgsSelectionProblem(BaseProblem):
    def __init__(self, dimension: int, **kwargs):
        program_id: int = np.random.choice(_candidate_program_index)
        assert dimension < len(_tot_opt_list)
        self.dimension = dimension
        super().__init__(self.dimension, **kwargs)
        self.program_dict = all_bench_program_list[program_id]
        assert self.program_dict["id"] == program_id + 1
        # no_conflict_ids = [index for index, opt in enumerate(_tot_opt_list) if len(opt["conflict_list"]) == 0]
        self.candidate_opt: List[int] = list(
            np.random.choice(range(len(_tot_opt_list)), size=self.dimension, replace=False))

    def get_opt(self, opt_selection):
        """
        根据01串获取编译序列
        """
        compile_flag = '-Os'
        for i in range(len(opt_selection)):
            conflict_list = _tot_opt_list[i]['conflict_list']
            conflict_flag = False
            for fid in conflict_list:
                if int(opt_selection[fid - 1]) == 1:
                    opt_selection[i] = 0
                    conflict_flag = True
            if conflict_flag:
                continue
            cur_flag = _tot_opt_list[i]['compile_flag'][opt_selection[i]]
            if cur_flag != '':
                compile_flag += ' ' + cur_flag
        return compile_flag

    @staticmethod
    def get_compile_size(program, compile_flag):
        """Test the compiled executable file size against the program name and compilation options"""
        temp_path = os.path.abspath(os.curdir)
        if not os.path.exists(proj_tmp_path):
            os.mkdir(proj_tmp_path)
        problem_dir = Path(proj_tmp_path, program)
        if not os.path.exists(problem_dir):
            shutil.copytree(str(Path(os.environ['HOME'],"CK/ctuning-programs/program", program)), str(problem_dir))
        tmp_dir = Path(problem_dir,
                       "tmp_{}_{}".format(str(int(time.time())),
                                          "".join(random.sample('zyxwvutsrqponmlkjihgfedcba', 10))))
        while os.path.exists(tmp_dir):
            tmp_dir = Path(problem_dir,
                           "tmp_{}_{}".format(str(int(time.time())),
                                              "".join(random.sample('zyxwvutsrqponmlkjihgfedcba', 10))))
        os.mkdir(tmp_dir)
        compile_result = ck.access({'action': "compile",
                                    'module_uoa': 'program',
                                    'data_uoa': program,
                                    'flags': compile_flag,
                                    'tmp_dir': str(tmp_dir),
                                    'out': '',
                                    'skip_clean_after': 'yes',
                                    'clean': 'no',
                                    'share': 'no'
                                    })
        os.chdir(temp_path)
        shutil.rmtree(tmp_dir)
        if compile_result['return'] > 0:
            print("Compile Failed!")
            return 1e15
        if compile_result['misc']['compilation_success'] == 'no':
            print("Compile Failed!")
            return 1e15

        return compile_result['characteristics']['obj_size']

    def evaluate(self, solution: Union[NpArray, List[int]]):
        opt_list = [0 for _ in _tot_opt_list]
        for index, value in enumerate(solution):
            opt_list[self.candidate_opt[index]] = int(value)
        compile_flag = self.get_opt(opt_list)
        program = self.program_dict['program']
        return -self.get_compile_size(program, compile_flag)


if __name__ == '__main__':
    np.random.seed(1088)
    solutions = [np.random.randint(2, size=186) for _ in range(2)]
    for id in _candidate_program_index:
        problem = CompilerArgsSelectionProblem(dimension=186, program_id=id)
        st = time.time()
        [problem.evaluate(solution) for solution in solutions]
        print(id, "\t", time.time() - st)
