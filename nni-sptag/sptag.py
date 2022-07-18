import SPTAG


class Sptag:

    def __init__(self, algo, metric):
        self._algo = str(algo)
        self._para = {}
        self._metric = {'angular': 'Cosine', 'euclidean': 'L2'}[metric]

    def fit(self, X, para=None):
        self._sptag = SPTAG.AnnIndex(self._algo, 'Float', X.shape[1])
        self._sptag.SetBuildParam("NumberOfThreads", '32', "Index")
        self._sptag.SetBuildParam("DistCalcMethod", self._metric, "Index")

        if para:
            self._para = para
            for k, v in para.items():
                self._sptag.SetBuildParam(k, str(v), "Index")

        self._sptag.Build(X, X.shape[0], False)

    def set_query_arguments(self, MaxCheck=8192):
        self._maxCheck = MaxCheck
        self._sptag.SetSearchParam("MaxCheck", str(self._maxCheck), "Index")

    def query(self, v, k):
        return self._sptag.Search(v, k)[0]

    def save(self, fn):
        self._sptag.Save(fn)

    def __str__(self):
        s = ''
        if self._para:
            s += ", " + ", ".join(
                [k + "=" + str(v) for k, v in self._para.items()])
        return 'Sptag(metric=%s, algo=%s, check=%d' % (
            self._metric, self._algo, self._maxCheck) + s + ')'
