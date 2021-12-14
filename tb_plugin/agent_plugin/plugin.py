import mimetypes
import os

from werkzeug import wrappers

from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.scalar import metadata

from .text import process_event

_TEXT_PLUGIN_NAME = 'text'
_PLUGIN_DIRECTORY_PATH_PART = "/data/plugin/agent/"
_DEFAULT_DOWNSAMPLING = 100


class AgentPlugin(base_plugin.TBPlugin):
    """Raw summary example plugin for TensorBoard."""

    plugin_name = 'AGENT'

    def __init__(self, context):
        """Instantiates AgentPlugin.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._data_provider = context.data_provider
        self._downsample_to = (context.sampling_hints or {}).get(
            _TEXT_PLUGIN_NAME, _DEFAULT_DOWNSAMPLING
        )
        self._version_checker = plugin_util._MetadataVersionChecker(
            data_kind='text',
            latest_known_version=0,
        )

    def get_plugin_apps(self):
        return {
            "/scalars": self.scalars_route,
            "/tags": self._serve_tags,
            "/static/*": self._serve_static_file,
        }

    def index_impl(self, ctx, experiment):
        mapping = self._data_provider.list_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=_TEXT_PLUGIN_NAME,
        )
        result = {run: [] for run in mapping}
        for (run, tag_to_content) in mapping.items():
            for (tag, metadatum) in tag_to_content.items():
                md = metadata.parse_plugin_metadata(metadatum.plugin_content)
                if not self._version_checker.ok(md.version, run, tag):
                    continue
                result[run].append(tag)
        return result

    @wrappers.Request.application
    def _serve_tags(self, request):
        """Serves run to tag info.

        Frontend clients can use the Multiplexer's run+tag structure to request data
        for a specific run+tag. Responds with a map of the form:
        {runName: [tagName, tagName, ...]}
        """
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        run_info = self.index_impl(ctx, experiment)

        return http_util.Respond(request, run_info, "application/json")

    @wrappers.Request.application
    def _serve_static_file(self, request):
        """Returns a resource file from the static asset directory.

        Requests from the frontend have a path in this form:
        /data/plugin/example_raw_scalars/static/foo
        This serves the appropriate asset: ./static/foo.

        Checks the normpath to guard against path traversal attacks.
        """
        static_path_part = request.path[len(_PLUGIN_DIRECTORY_PATH_PART):]
        resource_name = os.path.normpath(
            os.path.join(*static_path_part.split("/"))
        )
        if not resource_name.startswith("static" + os.path.sep):
            return http_util.Respond(
                request, "Not found", "text/plain", code=404
            )

        resource_path = os.path.join(os.path.dirname(__file__), resource_name)
        with open(resource_path, "rb") as read_file:
            mimetype = mimetypes.guess_type(resource_path)[0]
            return http_util.Respond(
                request, read_file.read(), content_type=mimetype
            )

    def is_active(self):
        """Returns whether there is relevant data for the plugin to process.

        When there are no runs with scalar data, TensorBoard will hide the plugin
        from the main navigation bar.
        """
        return True

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path="/static/index.js")

    def scalars_impl(self, ctx, experiment, tag, run):
        """Returns scalar data for the specified tag and run.

        For details on how to use tags and runs, see
        https://github.com/tensorflow/tensorboard#tags-giving-names-to-data

        Args:
          tag: string
          run: string

        Returns:
          A list of ScalarEvents - tuples containing 3 numbers describing entries in
          the data series.

        Raises:
          NotFoundError if there are no scalars data for provided `run` and
          `tag`.
        """
        all_text = self._data_provider.read_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=_TEXT_PLUGIN_NAME,
            downsample=self._downsample_to,
            run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]),
        )

        text = all_text.get(run, {}).get(tag, None)
        if text is None:
            return []
        return [
            process_event(d.wall_time, d.step, d.numpy, False)
            for d in text
        ]

    @wrappers.Request.application
    def scalars_route(self, request):
        """Given a tag and single run, return array of ScalarEvents."""
        tag = request.args.get("tag")
        run = request.args.get("run")
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        body = self.scalars_impl(ctx, experiment, tag, run)
        return http_util.Respond(request, body, "application/json")
