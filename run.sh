python3 -m benchmarks.run --config benchmarks/examples/iflow/iflow.fault.auto.benchmark.1.yaml
echo "benchmark 1 done (32)"
python3 -m benchmarks.run --config benchmarks/examples/iflow/iflow.fault.auto.benchmark.2.yaml
echo "benchmark 2 done (48)"
python3 -m benchmarks.run --config benchmarks/examples/iflow/iflow.fault.auto.benchmark.3.yaml
echo "benchmark 3 done (64)"
python3 -m benchmarks.run --config benchmarks/examples/iflow/iflow.fault.auto.benchmark.4.yaml
echo "benchmark 4 done (64, nofault)"
python3 -m benchmarks.run --config benchmarks/examples/iflow/iflow.fault.auto.benchmark.5.yaml
echo "benchmark 5 done (64, 2x fault)"
python3 -m benchmarks.run --config benchmarks/examples/iflow/iflow.fault.auto.benchmark.6.yaml
echo "benchmark 6 done (64, 2x slower llm)"
python3 -m benchmarks.run --config benchmarks/examples/iflow/iflow.fault.auto.benchmark.7.yaml
echo "benchmark 7 done (64, inspect_without_pause)"
python3 -m benchmarks.run --config benchmarks/examples/iflow/iflow.fault.auto.benchmark.8.yaml
echo "benchmark 8 done (16)"
python3 -m benchmarks.run --config benchmarks/examples/iflow/iflow.fault.auto.benchmark.9.yaml
echo "benchmark 9 done (96)"