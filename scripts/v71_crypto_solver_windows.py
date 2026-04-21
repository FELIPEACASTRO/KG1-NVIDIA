"""Windows-compatible copy of Tong's cryptarithm_deduce solver.

Retira signal.SIGALRM (Linux-only). Mesma logica interna (max_solutions=200).
"""
from collections import Counter


OPS = [
    lambda a, b: a + b,  # 0: add
    lambda a, b: abs(a - b),  # 1: abs_diff
    lambda a, b: a * b,  # 2: mul
    lambda a, b: a * 100 + b,  # 3: concat
    lambda a, b: b * 100 + a,  # 4: reverse concat
]


def num_to_digits(n):
    if n == 0:
        return (0,)
    d = []
    while n > 0:
        d.append(n % 10)
        n //= 10
    return tuple(reversed(d))


def is_concat(ex):
    s0, s1, _, s3, s4, rsyms = ex
    return rsyms == (s0, s1, s3, s4) or rsyms == (s3, s4, s0, s1)


class Solver:
    OP_NAMES = ["add", "abs_diff", "mul", "concat", "rev_concat"]

    def __init__(self, examples, query, unique=True):
        self.examples = examples
        self.query = query
        self.unique = unique
        self.mapping = {}
        self.used = set()
        self.op_assign = {}
        self.answers = Counter()
        self.answer_info = {}
        self.max_solutions = 200

    def solve(self):
        self._process(0)
        if self.answers:
            best, best_count = self.answers.most_common(1)[0]
            total = sum(self.answers.values())
            if not self.unique and total > 1 and best_count < total * 0.3:
                return None, ({}, {})
            return best, self.answer_info.get(best, ({}, {}))
        return None, ({}, {})

    def _process(self, idx):
        if len(self.answers) >= self.max_solutions:
            return
        if idx == len(self.examples):
            self._compute_query()
            return

        s0, s1, op_sym, s3, s4, rsyms = self.examples[idx]
        rlen = len(rsyms)

        feasible_ops = []
        if rlen <= 3:
            feasible_ops.append(0)
        if rlen <= 2:
            feasible_ops.append(1)
        if rlen <= 4:
            feasible_ops.append(2)
        if rlen == 4:
            feasible_ops.extend([3, 4])

        for d0 in self._vals(s0):
            n0 = self._assign(s0, d0)
            if n0 is None:
                continue
            for d1 in self._vals(s1):
                n1 = self._assign(s1, d1)
                if n1 is None:
                    continue
                lv = d0 * 10 + d1
                for d3 in self._vals(s3):
                    n3 = self._assign(s3, d3)
                    if n3 is None:
                        continue
                    for d4 in self._vals(s4):
                        n4 = self._assign(s4, d4)
                        if n4 is None:
                            continue
                        rv = d3 * 10 + d4

                        ops_to_try = (
                            [self.op_assign[op_sym]]
                            if op_sym in self.op_assign
                            else feasible_ops
                        )

                        for op_id in ops_to_try:
                            result_val = OPS[op_id](lv, rv)
                            if op_id >= 3:
                                if result_val < 0 or result_val >= 10000:
                                    continue
                                rd = (
                                    result_val // 1000,
                                    (result_val // 100) % 10,
                                    (result_val // 10) % 10,
                                    result_val % 10,
                                )
                            else:
                                rd = num_to_digits(result_val)
                            if len(rd) != rlen:
                                continue

                            assigns = []
                            ok = True
                            for rs, rdig in zip(rsyms, rd):
                                ns = self._assign(rs, rdig)
                                if ns is None:
                                    ok = False
                                    break
                                assigns.append((rs, ns))

                            if ok:
                                op_new = op_sym not in self.op_assign
                                if op_new:
                                    self.op_assign[op_sym] = op_id
                                self._process(idx + 1)
                                if op_new:
                                    del self.op_assign[op_sym]

                            for rs, ns in reversed(assigns):
                                self._undo(rs, ns)

                            if len(self.answers) >= self.max_solutions:
                                self._undo(s4, n4)
                                self._undo(s3, n3)
                                self._undo(s1, n1)
                                self._undo(s0, n0)
                                return

                        self._undo(s4, n4)
                    self._undo(s3, n3)
                self._undo(s1, n1)
            self._undo(s0, n0)

    def _vals(self, sym):
        if sym in self.mapping:
            return (self.mapping[sym],)
        if self.unique:
            return tuple(d for d in range(10) if d not in self.used)
        return range(10)

    def _assign(self, sym, dig):
        if sym in self.mapping:
            return False if self.mapping[sym] == dig else None
        if self.unique and dig in self.used:
            return None
        self.mapping[sym] = dig
        if self.unique:
            self.used.add(dig)
        return True

    def _undo(self, sym, was_new):
        if was_new is True:
            if self.unique:
                self.used.discard(self.mapping[sym])
            del self.mapping[sym]

    def _compute_query(self):
        qs0, qs1, qop, qs3, qs4 = self.query
        for s in (qs0, qs1, qs3, qs4):
            if s not in self.mapping:
                return

        ql = self.mapping[qs0] * 10 + self.mapping[qs1]
        qr = self.mapping[qs3] * 10 + self.mapping[qs4]
        if qop in self.op_assign:
            op_candidates = [self.op_assign[qop]]
        else:
            op_candidates = range(len(self.OP_NAMES))

        d2s = {}
        for s, d in self.mapping.items():
            if d not in d2s:
                d2s[d] = s

        for op_id in op_candidates:
            result_val = OPS[op_id](ql, qr)
            if op_id >= 3:
                if result_val < 0 or result_val >= 10000:
                    continue
                rd = (
                    result_val // 1000,
                    (result_val // 100) % 10,
                    (result_val // 10) % 10,
                    result_val % 10,
                )
            else:
                rd = num_to_digits(result_val)

            parts = []
            ok = True
            for d in rd:
                if d not in d2s:
                    ok = False
                    break
                parts.append(d2s[d])
            if not ok:
                continue

            ans = "".join(parts)
            self.answers[ans] += 1
            if ans not in self.answer_info:
                op_info = {k: self.OP_NAMES[v] for k, v in self.op_assign.items()}
                op_info[qop] = self.OP_NAMES[op_id]
                self.answer_info[ans] = (
                    dict(self.mapping),
                    op_info,
                )


def solve_problem(data):
    examples = []
    for e in data["examples"]:
        inp = e["input_value"]
        out = e["output_value"]
        examples.append((inp[0], inp[1], inp[2], inp[3], inp[4], tuple(out)))

    q = data["question"]
    query = (q[0], q[1], q[2], q[3], q[4])

    concat_ops = set()
    nonconcat_ops = set()
    for ex in examples:
        if is_concat(ex):
            concat_ops.add(ex[2])
        else:
            nonconcat_ops.add(ex[2])

    q_op = query[2]

    if q_op in concat_ops and q_op not in nonconcat_ops:
        for ex in examples:
            if ex[2] == q_op and is_concat(ex):
                s0, s1, _, s3, s4, rsyms = ex
                if rsyms == (s0, s1, s3, s4):
                    return query[0] + query[1] + query[3] + query[4], (
                        {},
                        {q_op: "concat"},
                    )
                else:
                    return query[3] + query[4] + query[0] + query[1], (
                        {},
                        {q_op: "rev_concat"},
                    )
        return query[0] + query[1] + query[3] + query[4], ({}, {q_op: "concat"})

    arith_examples = [ex for ex in examples if not is_concat(ex)]

    solver = Solver(arith_examples, query, unique=True)
    ans, info = solver.solve()
    if ans is not None:
        return ans, info

    solver2 = Solver(arith_examples, query, unique=False)
    return solver2.solve()
