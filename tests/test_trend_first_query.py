import ast
import asyncio
import gzip
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from starlette.responses import JSONResponse
from src.app.sensor_differences import difference_series
from src.app.trend_transport import TrendResponseCompression


class FirstTrendQueryTests(unittest.TestCase):
    def test_vectorized_statistics_match_finite_scalar_definition(self):
        rng = np.random.default_rng(11)
        # Repeated steps, excluded zero, nonfinite channels and large offsets.
        step = np.tile([0, 1, 2, 1, 3, 5, 5], 20)
        values = rng.normal(size=(len(step), 8))
        values[:, 0] += 1e9
        values[:, 1] = 4
        values[::3, 2] = np.nan
        values[:, 3] = np.inf
        values[::4, 4] = -np.inf
        names = ['Sensor'+str(i) for i in range(8)]
        row = difference_series(dict(actual=values, step=step, meta={'sensor_cols': names}),
                                pairs=[['Sensor0', 'Sensor1'], ['Sensor2', 'Sensor4']])
        for j, sensor in enumerate(row['sensor_series']):
            for st in [1, 2, 3, 5]:
                x = values[step == st, j]
                x = x[np.isfinite(x)]
                if not x.size:
                    self.assertNotIn(str(st), sensor['step_stats'])
                    continue
                stat = sensor['step_stats'][str(st)]
                self.assertEqual(stat['count'], x.size)
                for key, expected in dict(mean=x.mean(), min=x.min(), max=x.max(), std=x.std(ddof=0)).items():
                    self.assertEqual(stat[key], float(expected))
        self.assertNotIn('0', row['sensor_series'][0]['step_stats'])
        self.assertEqual(row['series'][-2]['step_stats']['1']['mean'],
                         float((values[step == 1, 0]-values[step == 1, 1]).mean()))

    def test_downsample_and_empty_sensor_inputs(self):
        data = np.arange(36).reshape(12, 3)
        step = np.repeat([1, 2], 6)
        row = difference_series(dict(actual=data, step=step,
                                    meta={'sensor_cols':['A','B','C'], 'interp_factor':2}))
        self.assertEqual(row['sensor_series'][0]['step_stats']['1']['mean'], 6.)
        self.assertEqual(row['sensor_series'][0]['step_stats']['1']['count'], 3)
        row = difference_series(dict(actual=np.empty((0, 0)), step=np.array([], dtype=int), meta={}))
        self.assertEqual(row['sensor_series'], [])

    def test_http_wrappers_return_native_json_without_schema_traversal(self):
        tree = ast.parse(Path('src/app/server.py').read_text())
        payload = {'model_id':'m','wafers':[{'missing':None,'mean':1.23456789012345,'name':'온도'}]}
        ns = dict(app=SimpleNamespace(post=lambda *a,**k:lambda f:f), JSONResponse=JSONResponse,
                  ScatterDifferencesReq=object, ScatterSensorsReq=object,
                  api_scatter_differences=lambda req:payload, api_scatter_step_times=lambda req:payload)
        for name in ['api_scatter_differences_http','api_scatter_step_times_http']:
            node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[node],type_ignores=[]), '<response>', 'exec'), ns)
            response = ns[name](object())
            self.assertIsInstance(response, JSONResponse)
            self.assertEqual(json.loads(response.body), payload)

    def test_gzip_only_for_trends_and_supported_clients(self):
        payload = {'wafers':[{'wid':str(i),'mean':1.23456789,'sensor':'온도'} for i in range(300)]}
        async def endpoint(scope, receive, send):
            await JSONResponse(payload)(scope, receive, send)
        app = TrendResponseCompression(endpoint)
        async def request(path, encoding):
            messages = []
            async def receive():return {'type':'http.request','body':b'','more_body':False}
            async def send(message):messages.append(message)
            await app({'type':'http','method':'POST','path':path,'headers':[(b'accept-encoding',encoding)]}, receive, send)
            headers = dict(messages[0]['headers'])
            body = b''.join(m.get('body',b'') for m in messages[1:])
            return headers, body
        for path in ['/api/scatter/differences','/api/scatter/step-times']:
            headers, body = asyncio.run(request(path,b'gzip'))
            self.assertEqual(headers[b'content-encoding'], b'gzip')
            self.assertIn(b'Accept-Encoding', headers[b'vary'])
            self.assertEqual(json.loads(gzip.decompress(body)), payload)
            self.assertLess(len(body),len(gzip.decompress(body)))
            headers, body = asyncio.run(request(path,b'identity'))
            self.assertNotIn(b'content-encoding',headers)
            self.assertEqual(json.loads(body),payload)
        headers, body = asyncio.run(request('/api/status',b'gzip'))
        self.assertNotIn(b'content-encoding',headers)
        self.assertEqual(json.loads(body),payload)
