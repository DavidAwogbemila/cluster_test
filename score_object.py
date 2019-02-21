from json import JSONEncoder

class ScoreObject():
    def __init__(self, score=None, mu_increment=None, sigma=None):
        self.score = score
        self.mu_increment = mu_increment
        self.sigma = sigma

class ScoreEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

