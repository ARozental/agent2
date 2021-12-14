import mimetypes
import json
import os

import werkzeug
from werkzeug import exceptions, wrappers

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
    headers = [('X-Content-Type-Options', 'nosniff')]

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
            '/index.js': self.static_file_route,
            '/index.html': self.static_file_route,
            '/scalars': self.scalars_route,
            '/runs': self.runs_route,
        }

    @staticmethod
    def respond_as_json(obj):
        content = json.dumps(obj)
        return werkzeug.Response(content, content_type='application/json', headers=AgentPlugin.headers)

    @wrappers.Request.application
    def static_file_route(self, request):
        filename = os.path.basename(request.path)
        extension = os.path.splitext(filename)[1]
        if extension == '.html':
            mimetype = 'text/html'
        elif extension == '.css':
            mimetype = 'text/css'
        elif extension == '.js':
            mimetype = 'application/javascript'
        else:
            mimetype = 'application/octet-stream'
        filepath = os.path.join(os.path.dirname(__file__), 'static', filename)
        try:
            with open(filepath, 'rb') as infile:
                contents = infile.read()
        except IOError:
            raise exceptions.NotFound("404 Not Found")
        return werkzeug.Response(
            contents, content_type=mimetype, headers=AgentPlugin.headers
        )

    def is_active(self):
        """Returns whether there is relevant data for the plugin to process.

        When there are no runs with scalar data, TensorBoard will hide the plugin
        from the main navigation bar.
        """
        return True

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path="/index.js")

    def _get_runs(self, ctx, experiment):
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
    def runs_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        return self.respond_as_json(self._get_runs(ctx, experiment))

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
