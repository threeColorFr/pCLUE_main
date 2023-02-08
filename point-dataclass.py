from dataclasses import dataclass, field, fields

@dataclass
class A:
    a: int = 1
    b: str = field(default='123',
        metadata = {'help': 'this is a test'}
    )
print(fields(A()))
print(fields(A())[1].metadata)
