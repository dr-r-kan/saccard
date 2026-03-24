from saccard import saccard


def _assert_common_result_contract(result, method: str):
    assert 'bpm' in result and method in result['bpm']
    assert 'times' in result
    assert 'metadata' in result
    assert result['fps'] > 0

    bpm = result['bpm'][method]
    times = result['times']
    assert len(bpm) > 0
    assert len(times) > 0
    assert result['metadata'].get('methods') is not None

    finite_bpm = [v for v in bpm if v == v]
    assert len(finite_bpm) > 0, f"No finite BPM values for method: {method}"


def run_smoke_test():
    print("Running unit tests...")

    methods = [
        'cpu_CHROM',
        'cpu_LGI',
        'cpu_POS',
        'cpu_PBV',
        'cpu_GREEN',
        'cpu_OMIT',
        'cpu_ICA',
        'cpu_SSR',
        'cpu_PCA',
    ]

    for method in methods:
        print(f"Testing method: {method}")
        result = saccard(
            'test.mp4',
            methods=[method],
            winsize=10,
            roi_method='convexhull',
            verb=False,
        )

        _assert_common_result_contract(result, method)
        print(f"Method {method} passed (windows={len(result['times'])})")

    print("All tests passed!")


if __name__ == '__main__':
    run_smoke_test()