class Response:
    def __init__(self, status, body):
        self.success = status
        self.body = body

def success(body=''):
    return Response(status=True, body=body)

def error(body=''):
    return Response(status=False, body=body)

def bool_response(boolean):
    return success('true') if boolean is True else success('false')

def serialize(gtp_command, gtp_response):
    return '{}{} {}\n\n'.format(
        '=' if gtp_response.success else '?',
        '' if gtp_command.sequence is None else str(gtp_command.sequence),
        gtp_response.body
    )
