import argparse
import sys
import time
import logging

# ---------------------------------------------------------
# Import Solvers
# ---------------------------------------------------------
try:
    from Baseline.ALNS import run_alns
except ImportError:
    run_alns = None

try:
    from Baseline.IGA import run_iga
except ImportError:
    run_iga = None

try:
    from Baseline.DABC import run_dabc
except ImportError:
    run_dabc = None

try:
    from Baseline.DIWO import run_diwo
except ImportError:
    run_diwo = None

try:
    from Baseline.Gurobi import buildModel as run_gurobi
except ImportError:
    run_gurobi = None


# ---------------------------------------------------------
# Main Dispatcher
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified Solver for AHASP (ALNS, IGA, DABC, DIWO, Gurobi)")

    # 1. Algorithm Selection
    parser.add_argument(
        '--algo',
        type=str,
        # required=True,
        choices=['ALNS', 'IGA', 'DABC', 'DIWO', 'Gurobi'],
        help="The algorithm to run."
    )

    # 2. Common Parameters
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help="Path to the instance Excel file."
    )
    parser.add_argument(
        '--time',
        type=int,
        default=3600,
        help="Time limit in seconds (default: 3600)."
    )

    # 3. Heuristic Specific Parameters
    parser.add_argument(
        '--iter',
        type=int,
        default=100,
        help="Maximum number of iterations (Heuristics only)."
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Execution Logic
    # -----------------------------------------------------
    print("\n" + "=" * 60)
    print(f" AHASP UNIFIED SOLVER")
    print("=" * 60)
    print(f" Algorithm   : {args.algo}")
    print(f" Instance    : {args.file}")
    print(f" Time Limit  : {args.time}s")
    if args.algo != 'Gurobi':
        print(f" Iterations  : {args.iter}")
    print("=" * 60 + "\n")

    start_time = time.time()
    result_summary = {}

    # --- Dispatch ---
    try:
        if args.algo == 'ALNS':
            if run_alns is None: raise ImportError("ALNS.py not found.")
            best_sol, best_fit, elapsed = run_alns(args.file, args.iter, args.time)
            result_summary = {'Fitness': best_fit, 'Time': elapsed, 'Type': 'Heuristic'}

        elif args.algo == 'IGA':
            if run_iga is None: raise ImportError("IGA.py not found.")
            best_sol, best_fit, elapsed = run_iga(args.file, args.iter, args.time)
            result_summary = {'Fitness': best_fit, 'Time': elapsed, 'Type': 'Heuristic'}

        elif args.algo == 'DABC':
            if run_dabc is None: raise ImportError("DABC.py not found.")
            best_sol, best_fit, elapsed = run_dabc(args.file, args.iter, args.time)
            result_summary = {'Fitness': best_fit, 'Time': elapsed, 'Type': 'Heuristic'}

        elif args.algo == 'DIWO':
            if run_diwo is None: raise ImportError("DIWO.py not found.")
            best_sol, best_fit, elapsed = run_diwo(args.file, args.iter, args.time)
            result_summary = {'Fitness': best_fit, 'Time': elapsed, 'Type': 'Heuristic'}

        elif args.algo == 'Gurobi':
            if run_gurobi is None: raise ImportError("Gurobi.py not found.")
            # Gurobi returns a dict or None
            g_res = run_gurobi(args.file, time_limit=args.time)
            if g_res:
                result_summary = {
                    'Fitness': g_res['objective'],
                    'Time': time.time() - start_time,  # Approximation if not returned
                    'Type': 'Exact',
                    'Status': g_res['status']
                }
            else:
                result_summary = {'Fitness': float('inf'), 'Time': args.time, 'Type': 'Exact', 'Status': 'Infeasible'}

    except Exception as e:
        print(f"\n[!] CRITICAL ERROR executing {args.algo}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # -----------------------------------------------------
    # Final Report
    # -----------------------------------------------------
    print("\n" + "#" * 60)
    print(" FINAL EXECUTION SUMMARY")
    print("#" * 60)
    print(f" Algorithm    : {args.algo}")

    if result_summary['Type'] == 'Heuristic':
        print(f" Best Fitness : {result_summary['Fitness']:.4f}")
        print(f" Elapsed Time : {result_summary['Time']:.4f}s")
    else:
        # Exact method output
        print(f" Objective    : {result_summary['Fitness']:.4f}")
        print(f" Status Code  : {result_summary.get('Status', 'N/A')}")
        print(f" Elapsed Time : {result_summary['Time']:.4f}s")

    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()