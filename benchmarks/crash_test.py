#!/usr/bin/env python3
"""
Transactional Integrity Test ("Jepsen-lite")
Verifies that ToonDB remains consistent after a process crash (kill -9).

Scenario:
1. Writer process continually increments a counter in the DB.
2. Main process acts as Chaos Monkey and kills the writer via SIGKILL.
3. Main process restarts DB and verifies the counter value is valid (atomic write).
"""

import os
import sys
import time
import signal
import shutil
import multiprocessing
import argparse
from typing import Optional

try:
    import toondb
except ImportError:
    print("ToonDB not installed. Skipping crash test.")
    sys.exit(0)

DB_PATH = "/tmp/toondb_crash_test"
COUNTER_KEY = b"crash_counter"

def writer_process(path: str, interval: float = 0.001):
    """Continuously updates a counter in the DB."""
    try:
        db = toondb.Database.open(path)
        i = 0
        print(f"[Writer] Started at {path}")
        while True:
            # Atomic PUT
            # In a real WAL enabled DB, this should be atomic
            # We write a value that we can validate (integer)
            val = str(i).encode('utf-8')
            db.put(COUNTER_KEY, val)
            
            if i % 1000 == 0:
                print(f"[Writer] count={i}", flush=True)
            
            i += 1
            if interval > 0:
                time.sleep(interval)
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[Writer] Failed: {e}")
        sys.exit(1)

def run_crash_test():
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
    
    # 1. Start Writer
    p = multiprocessing.Process(target=writer_process, args=(DB_PATH, 0.0))
    p.start()
    
    print(f"[Chaos] Writer started (PID={p.pid}). Letting it run for 2s...")
    time.sleep(2.0)
    
    # 2. KILL -9
    print(f"[Chaos] Simulating power failure (KILL -9)...")
    os.kill(p.pid, signal.SIGKILL)
    p.join() # Wait for it to be dead
    
    print(f"[Chaos] Process killed. Verifying consistency...")
    
    # 3. Recovery & Verification
    try:
        # Re-open DB. This should trigger WAL recovery if implemented.
        start_time = time.perf_counter()
        db = toondb.Database.open(DB_PATH)
        recovery_time = (time.perf_counter() - start_time) * 1000
        
        val = db.get(COUNTER_KEY)
        
        if val is None:
            print("[FAIL] Counter key lost!")
            return False
            
        try:
            count = int(val.decode('utf-8'))
            print(f"[PASS] Recovered successfully!")
            print(f"       Last Committed Value: {count}")
            print(f"       Recovery Time: {recovery_time:.2f}ms")
            return True
        except ValueError:
            print(f"[FAIL] Data corruption! Value is not an int: {val}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Open failed during recovery: {e}")
        return False
    finally:
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)

if __name__ == "__main__":
    print("="*60)
    print("TRANSACTIONAL INTEGRITY TEST (Crash Consistency)")
    print("="*60)
    
    success = run_crash_test()
    
    if success:
        print("\n✅ SYSTEM PASSED CRASH TEST")
        sys.exit(0)
    else:
        print("\n❌ SYSTEM FAILED CRASH TEST")
        sys.exit(1)
