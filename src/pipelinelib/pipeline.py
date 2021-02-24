import os
from itertools import filterfalse
from typing import Dict, List, Set

import pandas as pd
from matplotlib.figure import Figure

from src.pipelinelib.adapter import Adapter
from src.pipelinelib.querying import Queryable

from .component import Component
from .extension import Extension


class Pipeline:
    """
    A class to represent a pipeline for training a model

    Attributes
    ----------
    _queryable: Queryable
        convenience class to query DataFrames containing the text to train,
        in addition to other metadata

    _components: List[Component]
        all pipeline steps that will be executed in storage order

    _extensions: Set[Extension]
        a collection of extensions that the pipeline will create
        within the lookup structure and associate DataFrames with
    """

    def __init__(self, queryable: Queryable, empty_pipeline=False):
        self._queryable = queryable
        self._extensions: Set[Extension] = set()
        self._components: List[Component] = list()

        self._plot_output = "./images"

        if empty_pipeline:
            self._remove_all_pipes()

    def add_component(self, component: Component) -> "Pipeline":
        """
        Add a component to the pipeline, with checks
        """
        self._is_compatible(component)
        self._register_extensions(component)
        self._register_pipe(component)

        return self

    def add_components(self, components: List[Component]) -> "Pipeline":
        """
        Assemble components for features to the pipeline
        """
        for component in components:
            self.add_component(component)

        return self

    def execute(self, visualise: bool = False) -> Dict[Extension, pd.DataFrame]:
        """
        Execute the pipeline with the registered components
        """
        pipe_names = [component.name for component in self._components]
        print(f"=== Starting pipeline with {pipe_names} ===")

        curr: Dict[Extension, pd.DataFrame] = dict()
        ext_plot_map = dict()

        for component in self._components:
            result = component._internal_apply(
                storage=curr, queryable=self._queryable)

            missing_exts = [e for e in component.creates_extensions if e not in result]
            if missing_exts:
                names = [e.name for e in missing_exts]
                raise AssertionError(
                    f"{component.name} did not create required extension(s): {names}")

            not_valid = [e for e in component.creates_extensions
                         if result.get(e, None) is None]
            if not_valid:
                names = [e.name for e in not_valid]
                raise AssertionError(
                    f"{component.name} created {names}, but set None as their value!")

            if result and visualise:
                plots = component.visualise(result, self._queryable)
                self._store_plots(component, plots)
                ext_plot_map.update(plots)

                # ext_plot_map.update(plots)

            if result:
                for extension, df in result.items():
                    extension.store_to(curr, df)

        print("=== Finished pipeline execution ===")
        return (curr, ext_plot_map) if visualise else curr


    def _is_compatible(self, component: Component):
        """
        Check that the component will not overwrite a preregistered extension,
        or if it depends on an extension that has not been declared yet
        """

        # depends on non-existent Extension
        missing_extensions = list(filterfalse(
            self._extensions.__contains__, component.required_extensions))
        if missing_extensions:
            extension_names = map(lambda e: e.name, missing_extensions)
            raise Exception(
                f"Unable to apply {component.name} to pipeline - missing extensions: " +
                f"{', '.join(extension_names)}")

        # Exception case for adapter: should be allowed to overwrite fields
        if not isinstance(component, Adapter):
            # would overwrite pre-existing Extensions
            overwritten_extensions = list(filter(
                self._extensions.__contains__, component.creates_extensions))
            if overwritten_extensions:
                extension_names = map(lambda e: e.name, overwritten_extensions)
                raise Exception(
                    f"Unable to apply {component.name} to pipeline: - would overwrite extensions: "
                    + f"{', '.join(extension_names)}")

    def _register_extensions(self, component: Component):
        """
        Register extensions declared in component
        """
        self._extensions.update(component.creates_extensions)

    def _register_pipe(self, component: Component):
        """
        Add the transformation declared by the component to the pipeline
        """
        self._components.append(component)

    def _remove_pipe(self, component: Component):
        # Compare by name, grab index
        names = enumerate(map(lambda c: c.name, self._components))
        with_same_names = filter(
            lambda index_name: index_name[1] == component.name, names)
        index, _ = next(with_same_names, default=(None, None))

        # Not found, don't delete
        if not index:
            return

        # Remove from list
        self._components.pop(index)

    def _remove_all_pipes(self):
        self._components.clear()

    def _store_plots(self, component: Component, plots: Dict[Extension, Figure]):
        if not os.path.exists(self._plot_output):
            os.mkdir(self._plot_output)

        for ext in plots:
            path = os.path.join(self._plot_output, ext.filename())
            plots[ext].savefig(path)

    def __repr__(self) -> str:
        return f"{Pipeline.__name__}: {self._components}"
