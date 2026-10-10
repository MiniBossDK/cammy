from cammy.schedule import due


def test_every_one_always_runs():
    assert all(due(i, 1) for i in range(10))
    assert all(due(i, 0) for i in range(10))


def test_every_two_alternates():
    assert [due(i, 2) for i in range(6)] == [True, False, True, False, True, False]


def test_offset_staggers_two_detectors():
    for i in range(10):
        assert due(i, 2, 0) != due(i, 2, 1)


def test_every_three_runs_once_in_three():
    assert sum(due(i, 3, 1) for i in range(9)) == 3
