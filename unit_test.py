from saccard import saccard


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

        assert 'bpm' in result and method in result['bpm']
        assert len(result['bpm'][method]) > 0
        assert result['fps'] > 0
        print(f"Method {method} passed (windows={len(result['times'])})")

    print("All tests passed!")


if __name__ == '__main__':
    run_smoke_test()