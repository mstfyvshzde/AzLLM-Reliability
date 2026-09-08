.PHONY: test test-fast status hashes

test:
	pytest -q

test-fast:
	pytest -q --disable-warnings --maxfail=1

status:
	git status

hashes:
	shasum -a 256 \
		data/processed/final/benchmark_v1.0_manifest.json \
		data/processed/adaptation/adaptation_v2.0_manifest.json \
		data/results/adaptation_pilot_v2.0/final_results_manifest.json
