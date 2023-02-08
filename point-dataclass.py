from dataclasses import dataclass, field, fields

@dataclass
class A:
    a: int = 1
    b: str = field(default='123',
        metadata = {'help': 'this is a test'}
    )
print(fields(A()))
print(fields(A())[1].metadata)


'''
(Field(name='a',type=<class 'int'>,default=1,default_factory=<dataclasses._MISSING_TYPE object at 0x7f30e14cd400>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({}),_field_type=_FIELD), Field(name='b',type=<class 'str'>,default='123',default_factory=<dataclasses._MISSING_TYPE object at 0x7f30e14cd400>,init=True,repr=True,hash=None,compare=True,metadata=mappingproxy({'help': 'this is a test'}),_field_type=_FIELD))
{'help': 'this is a test'}
'''
