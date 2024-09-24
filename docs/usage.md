# Usage

Without any configuration needed, you can execute
```
kedro run --pipeline prepare
```
followed by
```
kedro run
```
and a default pipeline should run. In this and following examples the leading `poetry run` is omitted for simplicity.

Note that is required to always run the `prepare` pipeline in advance to any actual processing pipeline.
This is because of the current implementation relies on dynamically created nodes that are depending on the configuration and therefore requiring two separate pipeline executions.

In summary, the following pipelines exist:
- `prepare` : generates all possible combinations of configurations based on the current parameter set
- `measure` : performs the actual time measurement by executing experiments for each of the previously generated configurations with the ability to parallelize processing
- `combine` : gathers all the results from the `measure` pipeline and combines them into a single output dataset
- `visualize` : takes the combined experiment results and generates your plots

The `default` pipeline covers `measure`, `combine` and `visualize`.
You can run them separately by specifying the pipeline name.

This project can take advantage of multiprocessing to evaluate numerous combinations of *qubits*, *depths* and *shots* in parallel in the `measure` pipeline.
To use this, you should explicitly call the individual pipelines.
In summary the whole experiment will then look as follows:
```
kedro run --pipeline prepare
kedro run --pipeline measure --runner quafel.runner.Parallel
kedro run --pipeline combine
kedro run --pipeline visualize
```

Here, only the pipeline `measure` will utilize multiprocessing and the rest will run single process.
We recommend this approach since there is no advantage by running the other pipelines in parallel as well.
Of course, you can run the `measure` pipeline in a single process as well by omitting the `--runner` option.
If for some reason the execution of the `measure` pipeline gets interrupted, running the same pipeline again **without** running `prepare` will allow re-using previously generated artefacts.

For details on the output, see the [Data Structure Section](#floppy_disk-data-structure).

Note that if you want to re-run e.g. the `visualize` pipeline, you have to re-run the `prepare` pipeline as well.
This is because intermediate data containing information about the partitions is being deleted after the `visualize` pipeline of an experimant successfully ran.
This constraint will be removed in future releases.
