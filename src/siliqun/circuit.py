"""Circuit data model and deliberately small OpenQASM 3 input subset."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Iterable


_SINGLE_QUBIT_GATES = {"x", "y", "z", "h", "rx", "ry", "rz"}
_TWO_QUBIT_GATES = {"cx", "cz", "swap"}
_SUPPORTED_GATES = _SINGLE_QUBIT_GATES | _TWO_QUBIT_GATES


@dataclass(frozen=True)
class Gate:
    """One validated ideal gate instruction."""

    name: str
    targets: tuple[int, ...]
    parameters: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in _SUPPORTED_GATES:
            raise ValueError(f"unsupported gate: {self.name}")
        expected_targets = 1 if self.name in _SINGLE_QUBIT_GATES else 2
        if len(self.targets) != expected_targets:
            raise ValueError(f"{self.name} requires {expected_targets} targets")
        if any(not isinstance(index, int) or index < 0 for index in self.targets):
            raise ValueError("targets must be non-negative integers")
        expected_parameters = 1 if self.name in {"rx", "ry", "rz"} else 0
        if len(self.parameters) != expected_parameters:
            raise ValueError(f"{self.name} requires {expected_parameters} parameters")


@dataclass
class Circuit:
    """A finite circuit over the baseline ideal gate set."""

    n_qubits: int
    gates: list[Gate] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.n_qubits, int) or self.n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")

    def add(self, name: str, *targets: int, parameters: Iterable[float] = ()) -> "Circuit":
        gate = Gate(name=name.lower(), targets=tuple(targets), parameters=tuple(parameters))
        if any(target >= self.n_qubits for target in gate.targets):
            raise ValueError("gate target is outside the circuit")
        if len(set(gate.targets)) != len(gate.targets):
            raise ValueError("two-qubit gate targets must be distinct")
        self.gates.append(gate)
        return self

    def x(self, target: int) -> "Circuit":
        return self.add("x", target)

    def h(self, target: int) -> "Circuit":
        return self.add("h", target)

    def rx(self, theta: float, target: int) -> "Circuit":
        return self.add("rx", target, parameters=(float(theta),))

    def ry(self, theta: float, target: int) -> "Circuit":
        return self.add("ry", target, parameters=(float(theta),))

    def rz(self, theta: float, target: int) -> "Circuit":
        return self.add("rz", target, parameters=(float(theta),))

    def cx(self, control: int, target: int) -> "Circuit":
        return self.add("cx", control, target)

    def cz(self, control: int, target: int) -> "Circuit":
        return self.add("cz", control, target)

    def swap(self, left: int, right: int) -> "Circuit":
        return self.add("swap", left, right)

    @classmethod
    def from_openqasm3(cls, source: str) -> "Circuit":
        """Parse the documented static OpenQASM 3 subset.

        Supported declarations are ``OPENQASM 3.0`` and one ``qubit[n] name``
        register. Supported instructions are the baseline gate names. Includes,
        measurement, classical control, custom gates, and timing syntax are not
        supported by this intentionally narrow parser.
        """

        if not isinstance(source, str):
            raise TypeError("source must be a string")
        statements = []
        for raw in source.split(";"):
            text = re.sub(r"//.*", "", raw).strip()
            if text:
                statements.append(text)
        register_name = None
        circuit: Circuit | None = None
        for statement in statements:
            if re.fullmatch(r"OPENQASM\s+3(?:\.0)?", statement, re.IGNORECASE):
                continue
            if statement.startswith("include"):
                continue
            declaration = re.fullmatch(r"qubit\[(\d+)\]\s+([A-Za-z_]\w*)", statement)
            if declaration:
                if circuit is not None:
                    raise ValueError("only one qubit register is supported")
                circuit = cls(int(declaration.group(1)))
                register_name = declaration.group(2)
                continue
            if circuit is None or register_name is None:
                raise ValueError("declare a qubit register before instructions")
            match = re.fullmatch(
                r"([A-Za-z_]\w*)(?:\(([^)]*)\))?\s+(.+)", statement
            )
            if not match:
                raise ValueError(f"unrecognised statement: {statement}")
            name, parameter_text, target_text = match.groups()
            name = name.lower()
            if name not in _SUPPORTED_GATES:
                raise ValueError(f"unsupported OpenQASM instruction: {name}")
            target_tokens = [token.strip() for token in target_text.split(",")]
            targets = tuple(_parse_target(token, register_name) for token in target_tokens)
            parameters = () if parameter_text is None else (_safe_angle(parameter_text),)
            circuit.add(name, *targets, parameters=parameters)
        if circuit is None:
            raise ValueError("a qubit declaration is required")
        return circuit


def _parse_target(token: str, register_name: str) -> int:
    match = re.fullmatch(rf"{re.escape(register_name)}\[(\d+)\]", token)
    if not match:
        raise ValueError(f"invalid qubit target: {token}")
    return int(match.group(1))


def _safe_angle(expression: str) -> float:
    """Evaluate a numeric expression using only numbers, pi, and arithmetic."""

    tree = ast.parse(expression.strip(), mode="eval")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("angle expressions support only numbers, pi, +, -, *, and /")

    return visit(tree)
