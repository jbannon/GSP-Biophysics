from typing import List, Tuple, Dict, NamedTuple


def unpack_parameters(
	D:Dict
	):
	if len(D.values())>1:
		return tuple(D.values())
	else:
		return tuple(D.values())[0]

def make_filepath(
	components:List[str],
	extension:str = None
	) -> str:
	if extension is not None:
		extension = "."+extension if extension[0]!="." else extension
	else:
		extension = ""
	print(extension)
	components = [comp[:-1] if comp[-1]== "/" else comp for comp in components]
	fpath = "".join(["/".join(components),extension])
	return fpath



def star_echo(msg:str) -> None:
	star_string = "".join(["*"]*len(msg))
	print("".join(["\n",star_string,"\n",msg,"\n",star_string,"\n","\n"]))