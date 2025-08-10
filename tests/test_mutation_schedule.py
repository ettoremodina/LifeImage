"""Summary: Tests mutation schedule linear decay behavior."""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / 'src'
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from evolution.mutation import MutationSchedule


def test_schedule_decay():
    sched = MutationSchedule(0.5, 0.1, 100)
    assert sched.current_sigma(0) == 0.5
    mid = sched.current_sigma(50)
    assert 0.1 < mid < 0.5
    assert sched.current_sigma(100) == 0.1

if __name__ == '__main__':
    test_schedule_decay()
    print('ok')
