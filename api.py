import falcon
import json
from vowpal_platypus.daemon import daemon_predict

class APIResource:
    def on_get(self, req, resp):
        payload = []
        for query in req.query_string.split('&'):
            if '=' in query:
                query_dict = {}
                query_dict[query.split('=')[0]] = query.split('=')[1]
                payload.append(query_dict)
            else:
                payload.append(query)
        print '(Called with ' + str(payload) + '.)'
        pred = daemon_predict(4040, {'f': payload})
        resp.body = str(pred)
 
api = falcon.API()
api.add_route('/api', APIResource())
