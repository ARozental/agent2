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

    plugin_name = 'agent'
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
            '/runs': self.runs_route,
            '/data': self.data_route,
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

    def _get_runs(self, ctx, experiment, plugin_name):
        mapping = self._data_provider.list_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=plugin_name,
        )

        result = {run: [] for run in mapping}

        for (run, tag_to_content) in mapping.items():
            for (tag, metadatum) in tag_to_content.items():
                md = metadata.parse_plugin_metadata(metadatum.plugin_content)
                if plugin_name != self.plugin_name and not self._version_checker.ok(md.version, run, tag):
                    continue
                result[run].append(tag)

        return result

    @wrappers.Request.application
    def runs_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        runs_text = self._get_runs(ctx, experiment, _TEXT_PLUGIN_NAME)
        runs_agent = self._get_runs(ctx, experiment, self.plugin_name)
        run_keys = list(set(runs_text.keys()).union(set(runs_agent.keys())))
        return self.respond_as_json([
            {
                'id': key,
                'tags': {
                    'text': runs_text.get(key, []),
                    'agent': runs_agent.get(key, []),
                }
            }
            for key in run_keys])

    def _get_data(self, ctx, experiment, run):
        reconstructed = self._data_provider.read_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=_TEXT_PLUGIN_NAME,
            downsample=self._downsample_to,
            run_tag_filter=provider.RunTagFilter(runs=[run], tags=[
                'reconstructed/0/text_summary',
                'reconstructed/1/text_summary',
                'reconstructed_e/0/text_summary',
                'reconstructed_e/1/text_summary',
            ]),
        )

        reconstructed = reconstructed[run]
        for key, values in reconstructed.items():
            reconstructed[key] = [process_event(d.wall_time, d.step, d.numpy, False) for d in values]

        agent = self._data_provider.read_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=self.plugin_name,
            downsample=self._downsample_to,
            run_tag_filter=provider.RunTagFilter(runs=[run], tags=[
                'expected/1',
                'pndb/update_gate/1',
            ]),
        )

        if run in agent:
            agent = agent[run]
            agent['expected/1'] = [process_event(d.wall_time, d.step, d.numpy, False) for d in agent['expected/1']]
            agent['pndb/update_gate/1'] = [{
                'step': d.step,
                'wall_time': d.wall_time,
                'update_gate': d.numpy.tolist(),
            } for d in agent['pndb/update_gate/1']]
        else:
            agent = None

        result = []
        first_key = 'reconstructed/0/text_summary'
        for i in range(len(reconstructed[first_key])):
            result.append({
                'step': reconstructed[first_key][i]['step'],
                'wall_time': reconstructed[first_key][i]['wall_time'],
                'reconstructed': {key: item[i]['text'] for key, item in reconstructed.items()},
                'expected': None if agent is None else agent['expected/1'][i]['text'],
                'pndb': None if agent is None else {
                    'update_gate': agent['pndb/update_gate/1'][i]['update_gate'],
                },
            })

        return result

    @wrappers.Request.application
    def data_route(self, request):
        run = request.args.get('run_id')
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        return self.respond_as_json(self._get_data(ctx, experiment, run))
