'''utility funkshuns'''
from datetime import datetime
import errno
import pandas as pd
import numpy as np
from pathlib import PurePath,Path
import os
import shutil
from statistics import mean
import math
import json
import subprocess
printImportWarnings = False

regxASCII = "[^\W\d_]+"

def timeit(func):
    def wrapper(*args, **kwargs):
        nowe = datetime.now()
        print(f'Running {func.__name__}...\n \
            Started: {nowe.strftime("%H:%M")}\n') 
        ret = func(*args, **kwargs)
        timd = datetime.now()-nowe
        print(f'Finished in {timd}')
        return ret
    return wrapper
def prog(index, total,title='', bar_len=50 ):
    '''
    prints current progress to progress bar\n
    index is expected to be 0 based index. 
    0 <= index < total
    '''
    percent_done = (index)/total*100
    percent_done = np.ceil(percent_done*10)/10

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    
    emoji = '⏳' if (round(percent_done) < 100) else '✅'
    print(f'\t{emoji}{title}: [{done_str}{togo_str}] {percent_done}% done', end='\r')

import contextlib

@contextlib.contextmanager
def workingDir(path):
    """Changes working directory and returns to previous on exit."""
    #this violates the abstraction architecture a bit but whatever
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)
def cmder(cmd,*ARRGHHHSSSS):
    '''pass to cmd line and print output from py\n
    cmd: str 
    returns time elapsed, TODO allow returns from cmd\n
    TODO kwargs passed to cmd
    https://stackoverflow.com/a/6414278/13030053'''
    thenn = datetime.now()
    p = subprocess.Popen([cmd,*ARRGHHHSSSS], stdout=subprocess.PIPE, bufsize=1)
    for line in iter(p.stdout.readline, b''):
        print(line.split(b'>')[-1]),
    p.stdout.close()
    p.wait()

    elaps = datetime.now() - thenn
    return elaps

def fileBackup(filee,dir='Backup'):
    pthBackup = filee.parent/dir
    if not pthBackup.exists():
        pthBackup.mkdir()
    if not filee.exists():
        print(f'WARNING: {filee} doesnt exist to back up')
        return
    if filee.suffix =='.shp':
        filees = filee.parent.glob(f'{filee.stem}.*')
        [ shutil.copy(filee,pthBackup/filee.name) for filee in filees ]
    else:
        shutil.copy(filee,pthBackup/filee.name)
    print(f'{filee} backed up to {pthBackup}')
def fileRestore(filee,dir='Backup'):
    bkfilee = filee.parent/dir/filee.name
    tmpfilee = filee.parent/f'{filee.name}.tmp'
    
    shutil.move(bkfilee,tmpfilee)
    shutil.move(filee,bkfilee)
    shutil.move(tmpfilee,filee)

    print(f'Backup at \n{bkfilee}\nrestored to \n{filee}.\n\nThe two files have been swapped if you need to recover the previous {filee}')


def rem(path):
    """ param <path> could either be relative or absolute. """
    if path.exists():
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            shutil.rmtree(path)  # remove dir and all contains
    else:
        print('WARNING: Path ' + str(path) + ' does not exist')
def rems(paths,backup=True):
    '''backsup and deletes all in list of Paths if exists'''
    for filee in paths:
        if backup:
            if filee.exists():
                pthBackup = filee.parent/'Backup'
                if not pthBackup.exists():
                    pthBackup.mkdir()
                shutil.move(filee,pthBackup/filee.name)
            else:
                print('WARNING: Path ' + str(filee) + ' does not exist')
        else:
            rem(filee)
def fileExt(fle):
    '''returns the file extension (minus the '.') of a str or Path obj\n
    if neither, returns None (no error!)\n
    assert fileExt(myFile) == "txt" '''
    if type(fle) == str:
        from os.path import splitext
        ext = splitext(fle)[1].replace('.','')
    elif isinstance(fle, PurePath): #path object
        ext = fle.suffix.replace('.','')
    else:
        ext=None
    return ext
def listFiles(pth):
    '''List directory tree structure'''
    startpath = str(pth)
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
def getDirSize(pth = '.'):
    '''recursive, pth: Path or str'''
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(str(pth)):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size
def jsonAppend(jsonpth,dictt,indent=4):
    with open(jsonpth) as f:
        data = json.load(f)
    data.update(dictt)
    with open(jsonpth, 'w') as outfile:
        json.dump(data, outfile,indent=indent)
def xl2json(outpth,XLdoc,shtname):
    ''''''
    sht = pd.read_excel(XLdoc,sheet_name=shtname)
    sht = sht.dropna(how='all')
    sht = sht.T.reset_index().T #cols becomes top row
    jj = sht.values.tolist()
    # sht.to_json(pth/f'{shtname}.json',orient='table',index=False,indent=4)
    
    outjson = outpth/f'{shtname}.json'
    with open(outjson, 'w') as outfile:
        json.dump(jj, outfile,indent=4)
    
    outjson.write_text(outjson.read_text().replace('NaN','null'))
    print(f'{XLdoc} sheet {shtname} written to {outjson}')
    return outjson
class htmler:
    def decapitate(htmlpth):
        '''
        return (head,bod)
        '''
        starter = lambda haystack,needle: haystack.find(needle)+len(needle)
        h = htmlpth.read_text()

        #if there's meta/title, skip
        hstart = max( [starter(h,st) for st in ('<head>','<meta charset="utf-8">','</title>')] )
        hend = h.find('</head>',hstart)
        head = h[hstart:hend]
        assert len(head)>0

        bstart = h.find('<body>',hend)+len('<body>')
        bend = h.find('</body>',bstart)
        bod = h[bstart:bend]
        assert len(bod)>0
        return (head,bod)
    def concat(htmls,outhtml):
        '''
        Combines the heads/bodies of htmls: list of HTML Paths
        bounces to outhtml
        the first html in htmls will be the template (title, etc) and they will be concatenated in order
        YMMV but seems to work well for plotly/bokeh plots bounced from Python'''
        guts = [htmler.decapitate(htmlpth) for htmlpth in htmls]
        heads,bods = list(zip(*guts))
        res = htmls[0].read_text()
        assert res.find(heads[0])>0
        assert res.find(bods[0])>0
        res = res.replace(heads[0],'\n\n'.join(heads),1)
        res = res.replace(bods[0],'\n\n'.join(bods),1)
        outhtml.write_text(res)

        print(f'{htmls}\nconcatenated to\n{outhtml}')

    def ipyToHTML(ipy,outhtml,title=None,sitepth=None,
        bgcolor='#b0c6cf',stylee='default'):
        '''
        bounces jupyter notebook ipy to outhtml\n
        title replaces filename and html title\n
        if sitepth, save as sitepth/index.html too
        '''
        if not title:
            title=ipy.stem
        cmder('jupyter','nbconvert',str(ipy),'--no-input','--no-prompt','--to html')
        
        # rem blank cells
        tekst = outhtml.read_text(encoding='utf-8')
        tekst=tekst.replace('''<div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs jp-mod-noInput ">
        
        </div>''','')
        
        tekst=tekst.replace(f'<title>{ipy.stem}',f'<title>')
        
        tekst=tekst.replace('.dataframe','''    .dataframe {
        
            font-size: 15px !important;
            margin: auto !important;
            }
            .dataframe''',1)
        
        #unwhiten
        tekst=tekst.replace('#ffffff',bgcolor)

        if stylee=='default':
            stylee = '''
                <style>
                    h1 {
                    text-align:center;
                    }
                    body {
                        font-family: 'Roboto';
                        line-height: 1.3em;
                        background-color: '''+bgcolor+'''!important;
                    }
                    p {
                        padding:0%;
                        font-size:20px !important;
                        margin:2em 8% 0 8% !important;
                    }
                </style>'''
        
        key = '</style>'
        
        filee = outhtml
        putHere = tekst.rfind(key)
        filee.write_text(tekst[:putHere]+f'{key}\n\n'+stylee+tekst[putHere+len(key):] 
            ,encoding='utf-8')
        
        # tekst = outhtml.read_text(encoding='utf-8')
        # stylee='''    .dataframe {
        
        #       font-size: 18px !important;
        #       margin: auto !important;
        #     }
        #     .dataframe'''
        # key='.dataframe'
        
        # putHere = tekst.find(key)
        # filee.write_text(tekst[:putHere]+f'{key}\n\n'+stylee+tekst[putHere+len(key):] 
        #     ,encoding='utf-8')
        
        #rem plotly logo
        lines = txt.read(outhtml)
        key1 = '<div class="logo-block">'
        lines = lines[:txt.findKey(lines,key1)] + lines[txt.findKey(lines,[key1,'</div>'])+1:]
        
        txt.write(outhtml,lines)
        
        outhtml=shutil.move(outhtml,outhtml.parent/f'{title}.html')
        
        cmder(f"{str(outhtml)}")
        
        if sitepth:
            shutil.copy(outhtml,sitepth/'index.html')
            cmder('code',f"{str(sitepth)}")

        return outhtml
        
try:
    import requests
except Exception as e:
    if printImportWarnings:
        print('WARNING: ',e)


def DL(url,toPth,callback=None):
    '''DL's file from url and places in toPth Path\n
    returns request status code'''
    if not callback:
        print("downloading: ",url)
    if not toPth.exists():
        toPth.mkdir()
    # assumes that the last segment after the / represents the file name
    # if url is abc/xyz/file.txt, the file name will be file.txt
    file_name_start_pos = url.rfind("/") + 1
    file_name = url[file_name_start_pos:]

    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        with open(toPth/file_name, 'wb') as f:
            for data in r:
                f.write(data)
    if callback:
        callback()
    return r.status_code

from multiprocessing.pool import ThreadPool

@timeit
def DLmulti(urls,toPth,threds=12):
    toPth.mkdir(parents=True,exist_ok=True)
    #TODO starmap_async?
    results = ThreadPool(threds).starmap(DL, 
        [ 
            ( url,toPth,prog(i,len(urls),f'Downloading {url}') ) 
            for i,url in enumerate(urls) 
            ])
    return list(zip(urls,results))

from multiprocessing import Process,freeze_support

def runInParallel(funks):
    '''funks: [ ( func1,(arg1,arg2) ) , ( func2,(arg1,) ) ]\n
    uses multiprocessing.Process\n
    WARNING: Does NOT work from ipynb's'''    
    freeze_support()
    proc = []
    for fn,argz in funks:
        p = Process(target=fn,args=argz)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()
    return 'done'

import zipfile
import gzip
@timeit
def unzipAll(pthOfZips,to1pth=None):
    '''unzips all .zip's in pthOfZips to individual folders\n
    if to1pth: unzip all to Path to1pth instead'''

    for paf in pthOfZips.glob('*.gz'):
        if to1pth:
            to1pth.mkdir(parents=True,exist_ok=True)
            try:
                with gzip.open(paf, 'rb') as f_in:
                    with open(to1pth/paf.name.replace('.gz',''), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            except Exception as e:
                print('WARNING: ',e,paf)
        else:
            raise NotImplementedError("unzipAll .gz's to individual paths")
        
    for paf in pthOfZips.glob('*.zip'):
        print (paf)
        unzipd = paf.parent/paf.stem
        if to1pth:
            if not to1pth.exists():
                to1pth.mkdir()
            try:
                zyp = zipfile.ZipFile(paf)
                zyp.extractall(path=to1pth)
            except zipfile.BadZipfile as e:
                print(f'BAD ZIP: {paf}')
                try:
                    paf.unlink()
                except OSError as e: # this would be "except OSError, e:" before Python 2.6
                        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
                            raise # re-raise exception if a different error occured ```
        else:
            if not unzipd.exists():
                try:
                    zyp = zipfile.ZipFile(paf)
                    unzipd.mkdir()
                    zyp.extractall(path=unzipd)
                except zipfile.BadZipfile as e:
                    print(f'BAD ZIP: {paf}')
                    try:
                        paf.unlink()
                    except OSError as e: # this would be "except OSError, e:" before Python 2.6
                        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
                            raise # re-raise exception if a different error occured ```

def DLunzip(urls,toPth,threds=12):
    '''DLmulti then unzipAll
    handles .zip or .gz'''
    zipth = toPth/'zipd'
    res = DLmulti(urls,zipth)
    unzipAll(zipth,toPth)
    shutil.rmtree(zipth)
    return res

def transferAttrs(oneObj,anotherObj='new'):
    '''transfer all attrs from oneObj to anotherObj which don't start with _\n
    if anotherObj=='new': creates a dummy obj with the attrs\n
    \n
    returns anotherObj'''
    if anotherObj=='new':
        anotherObj =  lambda: None
    [ setattr( anotherObj,atr,getattr(oneObj,atr) ) 
        for atr in dir(oneObj) if not atr.startswith('_')]
    return anotherObj
def attrsToDict(obj,skipWithPrefix='_'):
    return {atr:getattr(obj,atr) for atr in dir(obj) if not atr.startswith(skipWithPrefix)}

def bold(text):
    '''for plots'''
    return f'<b>{text}</b>'

from functools import reduce
from operator import iconcat
def flattenList(listOfLists):
    '''or list of tups'''
    return reduce(iconcat,listOfLists,[])
def globdown(pth,extensh='.*',lvls=30):
    '''globs "recursively" down (lvls: int) levels\n
    extensh in the form ".ipynb"\n
    pth: Path'''
    return flattenList( [list(pth.glob(f'*{"/*"*lvl}{extensh}')) for lvl in range(lvls)] )
def strChunk(myString,chunkCharacters):
    '''split a string into chunks at regular intervals based on the number of characters in the chunk'''
    myString = myString.replace('\n','')
    return [myString[i:i+chunkCharacters] for i in range(0, len(myString), chunkCharacters)]
def listChunk(myList,n):
    n = max(1, n)
    return [myList[i:i+n] for i in list(range(0, len(myList), n)) ]
def listJoin(myList,delim='\n'):
    '''["a","b"] -> ["a","\n","b"] '''
    return sum([[i, delim] for i in myList[:]],[])[:-1]
# from copy import copy
def listInsertEveryNth(myList,N,insertWhat,allowInsertAtEnd=True):
    # return [x for y in (myList[i:i+N] + [insertWhat] * (i < len(myList) - (N-1)) for i in list(range(0, len(myList), N))) for x in y]
    lst = myList[:]
    i = N-1
    nd = 1 if allowInsertAtEnd else 0
    while i < len(lst)+nd:
        lst.insert(i, insertWhat)
        i += (N)
    return lst
def listDigits(myList,nDigits):
    '''[1,10,22,4] -> ['01', '10', '22', '04']'''
    return [str(i).rjust(2,'0') for i in myList]
def listisin(sublist,inlist):
    '''if sublist is subset to inlist'''
    return set(sublist).issubset( set(inlist) )
def assertListisin(sublist,inlist,lethal=True):
    condish = listisin(sublist,inlist)
    if not condish:
        msg = f'{[sub for sub in sublist if sub not in inlist]} not in {inlist}'
        if lethal:
            assert False, f'ERROR: {msg}'
        else:
            print(f'WARNING: {msg}')
def listrip(myList,val,stripFrom=[0,-1]):
    '''works like string.strip() but for lists\n
    strips trailing and leading items equal to value val\n
    this can be customized with stripFrom
    inplace and also returns'''
    for i in stripFrom:
        while myList[i]==val:
            myList.pop(i)
    return myList
def listEqual(alist):
    '''returns True if every item in alist iterable is equal'''
    return all(alist[i+1] == alist[0] for i in range(len(alist)-1))
try:
    from collections import Iterable
except:
    from collections.abc import Iterable #knuckleheads changing imports up between v's
def isiter(obj):
    '''True if iterable, False if string type, b'string', (geo)pandas obj, or not iterable'''
    return (
        isinstance(obj, Iterable) 
        and not isinstance(obj, str)
        and not isinstance(obj,bytes)
        and not isinstance(obj,pd.Series)
        and not isinstance(obj,pd.DataFrame)
    )
def asList(listOrThing):
    '''returns either copy of og list or list of str, depending on input type\n
    for validating args that can be single item or list\n
    newVar = asList(arg)\n
    Note: assign result to new name as it copies the incoming arg'''
    if isiter(listOrThing):
        return listOrThing[:]
    else:
        return [listOrThing]

def dropSuff(listOfStrs,n=1,delim=' '):
    '''drops the last n words of each str in a list, returns the new list (not inplace)'''
    spleet = [st.split(delim) for st in listOfStrs]
    dropt = [st[:-n] if len(st)>n else st[0] for st in spleet]
    #now could be str or a list, don't join a 'w o r d':
    return [delim.join(stlst) if isinstance(stlst,list) else stlst for stlst in dropt]

# def listify(maybelist): TODO for args
#     if (lst_cols is not None
#     and len(lst_cols) > 0
#     and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
#     lst_cols = [lst_cols]
class noTearsDict(dict):
    def __missing__(self, key):
        return key
def dictSwap(aDict):
    return {value:key for key, value in aDict.items()}

import re
def replaceMulti(adict, text):
    '''Note: happens in one pass!\n
    returns the new str'''
    # Create a regular expression from all of the dictionary keys
    regex = re.compile("|".join(map(re.escape, adict.keys(  ))))
    # For each match, look up the corresponding value in the dictionary
    return regex.sub(lambda match: adict[match.group(0)], text)
def ondah(namee):
    '''legalizes names for use in a SQL DB by adding a million underscores'''
    nm = namee[:]
    unsavoryCharacters = [' ','.','`','=','+','^',',']
    rep = dict(zip(unsavoryCharacters,['_']*len(unsavoryCharacters)))
    nm = replaceMulti(rep,nm)
    if nm[0].isdigit():
        nm = f'_{nm}'
    return nm
def extractDigits(strng):
    if type(strng)==str:
        res = re.findall('\d+',strng)[0]
    else:
        res = strng
    res = float(res)
    return res
def reNestedSub(strng,reggie,subFind,subRep,subflags=0):
    '''
    finds instances of reggie (regex obj)\n
    applies an inner sub(subFind,subRep,<current match>,flags=subflags) to each\n
    TODO subflags necessary if a compiled regx is passed?\n
    TODO understand if subFind, subRep are regex exprshs or just strs?
    '''
    adj = 0
    for match in re.finditer(reggie, strng):
        matcher = re.sub(subFind,subRep,match.groups()[0],subflags)
#         print(adj)
#         print(matcher)#re.sub(reggie, re.sub('\n','\nxxxx',r'\1'), tst)
#         print(match.span())
        strng = list(strng)
        strng[match.span()[0]+adj:match.span()[1]+adj] = matcher
        strng = ''.join(strng)
        adj += len(matcher) - len(match.groups()[0])
#         print(strng)
    return(strng)
def regx(findWhat,inWhat):
    '''just takes the boilerplate outta regex TODO remove mention of this, see wes mckinney book regex explanation'''
    comp = re.compile(findWhat)
    reggie = comp.search(inWhat)
    return reggie.group()
def extractNum(strng):
    '''given str, extracts first number values in str and returns it as a float\n
    given float, return the float'''
    if type(strng)==str:
        res = re.findall(r"[-+]?\d*\.\d+|\d+",strng)[0]
    else:
        res = strng
    res = float(res)
    return res
def rreplace(strng,oldStr,newStr):
    return newStr.join(strng.rsplit(oldStr, 1))
def findNth(findNthWhat,inWhat,n):
    '''inWhat.find() with new arg nth instance'''
    fnd = findNthWhat
    rep = 'X'*len(fnd)
    r = inWhat.replace(fnd, rep, n-1)
    return r.find(fnd)
import ctypes  # An included library with Python install.   
def mbox(text, title='', style=0):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def getinitboiler(args_st):
    s = [f'self.{st}' for st in args_st.split(',')]
    print(f'{",".join(s)} = {args_st}')
try:
    from inspect import getargspec
except: #py 3.11
    from inspect import getfullargspec
    getargspec=getfullargspec
# class autoinit(type):
#     '''Passes all args to self.\n
#     WARNING: doesn't work for kwargs\n
#     use:\n
#     class Person (metaclass=autoinit):
#         def __init__(self, first_name, last_name, birth_date, sex, address):
#         print('ok')
#         def __repr__(self):
#         return "{} {} {} {} {}"\
#         .format(self.first_name, self.last_name, self.birth_date, self.sex, self.address)
    
#     if __name__ == '__main__':
#         john = Person('Jonh', 'Doe', '21/06/1990', 'male', '216 Caledonia Street')
#         print(john)
#     https://www.codementor.io/@predo/self-initializing-classes-gro3q9svt'''
#     def __new__(meta, classname, supers, classdict):    
#         classdict['__init__'] = autoinit.autoInitDecorator(classdict['__init__'])
#         return type.__new__(meta, classname, supers, classdict)
#     def autoInitDecorator (toDecoreFun):
#         def wrapper(*args):

#             # ['self', 'first_name', 'last_name', 'birth_date', 'sex', 'address']
#             argsnames = getargspec(toDecoreFun)[0]

#             # the values provided when a new instance is created minus the 'self' reference
#             # ['Jonh', 'Doe', '21/06/1990', 'male', '216 Caledonia Street']
#             argsvalues = [x for x in args[1:]]

#             # 'self' -> the reference to the instance
#             objref = args[0]

#             # setting the attribute with the corrisponding values to the instance
#             # note I am skipping the 'self' reference
#             for x in argsnames[1:]:
#                 objref.__setattr__(x,argsvalues.pop(0))

#         return wrapper
from scipy.spatial import KDTree
class xnp():
    def ABtree(A,B,maxdist=2):
        '''literally just KDTree but return -1 for indices > maxdist'''
        dist,snap = KDTree(A).query(B,distance_upper_bound=maxdist)
        snap[dist>maxdist] = -1
        return snap
    def Atree(A,maxdist=2):
        '''KDtree of an array on itself, without matching a row to itself\n
        dist above maxdist returned as -1'''
        dist,snap = KDTree(A).query(A,k=2,distance_upper_bound=maxdist)
        snap[dist>maxdist] = -1
        dist
        snp = pd.DataFrame(snap,index=np.arange(len(snap)))
        snp
        snp = snp.replace(-1,np.nan)
        snp
        snp.loc[snp[0]==snp.index,0]=np.NaN
        snp.loc[snp[1]==snp.index,1]=np.NaN
        snp
        fild = snp[0].fillna(snp[1])
        fild
        snapped = fild.fillna(-1).astype(np.int64).to_numpy()
        snapped
        return snapped
    def KDTreee(A,B,maxdist=2):
        '''handles case where A and B are identical, in which case does a
        KDtree of an array on itself, without matching a row to itself\n
        returns -1 for indices > maxdist
        '''
        if (A==B).all():
            snap = xnp.Atree(A,maxdist)
        else:
            snap = xnp.ABtree(A,B,maxdist)
        return snap

    def find_nearest(array, value):
        '''NOT TESTED
        https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array'''
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    def sortByCol(an_array,colN):
        '''TODO by row'''
        return an_array[np.argsort(an_array[:, colN])]
    def replace(ar, dic,assume_all_present=False):
        '''with dict\n
        https://stackoverflow.com/a/47171600/13030053'''
        # Extract out keys and values
        k = np.array(list(dic.keys()))
        v = np.array(list(dic.values()))

        # Get argsort indices
        sidx = k.argsort()

        ks = k[sidx]
        vs = v[sidx]
        idx = np.searchsorted(ks,ar)

        if assume_all_present==0:
            idx[idx==len(vs)] = 0
            mask = ks[idx] == ar
            return np.where(mask, vs[idx], ar)
        else:
            return vs[idx]
from scipy.interpolate import interp1d
class xpd():
    '''Handy tools for pandas'''
    class prof(pd.Series):
        '''Extension of pd.Series where indexing interpolates
        linearly when the indexer is not exact
        \n 
        both index and value should be floats/ints'''
        def __getitem__(self,i):
            # TODO update func whenever values update
            # self.__getitem__ = interp1d(self.index.to_numpy(),self.to_numpy())
            f = interp1d(self.index.to_numpy(),self.to_numpy())
            return float(f(i))
    # class profDF(pd.DataFrame):
    #     '''Extension of pd.Series where indexing interpolates
    #     linearly when the indexer is not exact
    #     \n 
    #     both index and value should be floats/ints'''
    #     def __getitem__(self,i):
    #         # TODO update func whenever values update
    #         # self.__getitem__ = interp1d(self.index.to_numpy(),self.to_numpy())
    #         f = interp1d(self.index.to_numpy(),self.to_numpy())
    #         return float(f(i))

    # def alignIdxes(DFs):
    #     dfs = [df.copy() for df in DFs]
    #     TODO dfprof instead
    #     # convert all to series
    #     dfs = [ df[df.columns[0]] if isinstance(df,pd.DataFrame) 
    #         else df
    #             for df in dfs ]

    #     dfs = [xpd.prof()]
    #     crc = xpd.prof(crc.maxwsel)
    #     crc
    #     idx = pd.concat([dsc,vary,crc]).index.unique().sort_values().to_list()
    #     len(idx)
    #     dfs = [dsc,vary,crc]
    #     bnds = pd.DataFrame([ [df.index[0],df.index[-1]]
    #                         for df in dfs ],columns=['mn','mx']
    #                     )
    #     bnds
    #     minn,maxx = bnds.mn.max(),bnds.mx.min()
    #     minn,maxx
    #     idx = [ i for i in idx if i>minn and i<maxx ]
    #     len(idx)
    #     dfs = [
    #         pd.Series(dict(zip(idx,[ df[i] for i in idx ])))
    #         for df in dfs ]
    #     dfs[0]
    #     assert listEqual([df.index.to_list() for df in dfs])
    #     dsc,vary,crc = dfs
    #     100*(dsc.corr(crc) - vary.corr(crc))/vary.corr(crc)
    #     vary.corr(crc)
    #     bnds.shape
    #     xxr.pctBias(dsc.values,crc.values)
    #     xxr.pctBias(vary.values,crc.values)
    #     xxr.RMSE(dsc.values,crc.values)
    #     xxr.RMSE(vary.values,crc.values)

    def asDF(DForCSVorXLSX,**kwargs):
        '''returns DF from whatever input is\n
        if not already df,**kwargs get passed to pd.read_csv or pd.read_excel\n
        TODO hdf group'''
        df = DForCSVorXLSX
        if isinstance(df,pd.DataFrame):
            return df.copy()
        elif isinstance(df,PurePath):
            ext = df.suffix.replace('.','',1)
            if ext=='csv':
                return pd.read_csv(df,**kwargs)
            elif ext.startswith('x'):
                return pd.read_excel(df,**kwargs)
            elif ext == "tsv":
                return pd.read_csv(df, sep="\t",**kwargs)
            elif ext in ["parquet", "pq", "parq"]:
                return pd.read_parquet(df,**kwargs)
            else:
                raise NotImplementedError(f"Unsupported file type {ext}. Should be one of csv, tsv, parquet, or xlsx.")
        else:
            raise NotImplementedError(f"Unsupported type {type(df)}. Only file path / existing df supported at this time")
    
    def weightedAvg(df,w8col):
        '''returns the weighted average of cols in df grouped by df.index lvl 0\n
        w8col is the weighting to use for each item within each unique lvl0\n
        w8col and any indices beyond lvl 0 get dropped in the result\n\n
        
        lvl0 of df.index is the idx to aggregate by, 
        any further idx lvls will disappear in the aggregation\n
        ie preprocess with:\n
        df = df.set_index(['lvl0','lvl1']).sort_index()\n
        where there are multiple lvl1's per lvl0, and lvl1 is the idx of individual items to agg
        '''
        def _weightedAvg(df,w8col):
            # print('avg')
            tot = df.groupby(level=0).sum()[w8col]
            tot.name = 'tot'
            tot
            df = df.merge(tot,on=df.index.names[0])
            cols = df.columns[~df.columns.isin([w8col,'tot'])].to_list()

            df[cols] = df[cols].mul(df[w8col],axis=0).div(df['tot'],axis=0)
            df = df.groupby(level=0)[cols].sum()
            return df
        def _weightedAvgNaNs(df,w8col):
            '''slow af but handles NaNs correctly'''
            # print('slowwww')
            cols = df.columns.drop(w8col).to_list()

            def _maskAvg(grup):
                # print(grup,'\n')
                allna = grup.isna().all()
                nacols = allna.index[allna].to_list()
                
                grup = grup.dropna(how='all',axis=1)
                ccols = [c for c in cols if c not in nacols]
                data = np.ma.average(np.ma.array(grup[ccols].values, mask=grup[ccols].isnull().values), 
                        weights=grup[w8col].values, axis=0).data
                grup = grup.iloc[[0]]
                grup.loc[:,ccols] = data
                return grup
            grupd = df.groupby(level=0).apply(_maskAvg)
            grupd.index = grupd.index.get_level_values(0)
            return grupd[cols]
        
        if df[w8col].isna().any():
            raise NotImplementedError(f'to intelligently handle when w8col has nulls\n{df[df[w8col].isna()]}')

        if df.isna().any().any():
            grupd = _weightedAvgNaNs(df,w8col)
        else: #quicker
            grupd = _weightedAvg(df,w8col)

        minOK = df.drop(w8col,axis=1).min()<=grupd.min()+0.00001
        assert minOK.all(),f'the following cols got screwed up: \n{grupd.min()[~minOK]} > {df[~minOK].min()}'

        return grupd
    def lastNrowsEqual(df,n,conds=None):
        '''
        returns bool df of whether the last n rows have all been equal for each col\n
        conds: list of any additional bool df's you would like to check for\nex:\n
        nodata = lastNrowsEqual(gauge,5,conds = gauge!=gauge.min())\n
        gauge[~nodata]\n
        '''
        samesies = [ df.shift(i+1) == df.shift(i)
                                    for i in range(n) ]

        # & ( gauge.shift(2) == gauge.shift(1)) & (gauge.shift(1) == gauge)  & (gauge!=gauge.min())
        
        # nodata
        if conds is not None:
            samesies += asList(conds)

        df = pd.concat(samesies,axis=1,keys=range(len(samesies)))
        df
        df = df.swaplevel(axis=1)
        df
        df = df.T.sort_index().T
        df
        nodata = df.T.groupby(level=0).all().T
        return nodata
    def tsGaps(dtIndex_ts,freq='H'):
        '''    #The complete timeseries record needed\n
        fullts = pd.date_range(ts.min(),ts.max(),freq=freq)\n
        gaps = fullts[~fullts.isin(ts)]'''
        ts =dtIndex_ts
        #The complete timeseries record needed
        fullts = pd.date_range(ts.min(),ts.max(),freq=freq)
        gaps = fullts[~fullts.isin(ts)]
        return gaps
    def fromNestedDict(data):
        '''https://stackoverflow.com/questions/33611782/pandas-dataframe-from-nested-dictionary
        https://towardsdatascience.com/all-pandas-json-normalize-you-should-know-for-flattening-json-13eae1dfb7dd
        '''
        df_melt = pd.json_normalize(data, sep='>>').melt()
        
        #handle bug where it prefixes with sep, no change needed if they fix the bug
        df_melt.loc[df_melt['variable'].str.startswith('>>'),'variable'] = df_melt.loc[df_melt['variable'].str.startswith('>>'),'variable'].str[2:]
        
        df_final = df_melt['variable'].str.split('>>', expand=True)
        df_final.columns = [f'col{name}' for name in df_final.columns]
        df_final['value'] = df_melt['value']
        
        df=df_final
        df=df.set_index(df.columns[df.columns.str.contains('col')].to_list())
        
        return df
    def asNumeric(x):
        '''where possible\n
        df['A'].apply(xpd.asNumeric)'''
        try:
            return pd.to_numeric(x)
        except:
            return x
    def toXL(df,xldoc,**kwargs):
        savd = False
        while not savd:
            try:
                df.to_excel(xldoc,**kwargs)
                print(f'{df} successfully saved to \n{xldoc}')
                savd=True
            except:
                ans = mbox(f'Please close the Excel doc {xldoc.name} and hit <Retry> to continue. \n\n<Abort> will close the program without exporting to Excel.',style=2)
                if ans == 3:
                    print(f'Aborting, {xldoc} not successfully written')
                    savd = True
    def fixgroupby(df):
        '''groupby(level=lvl).apply(func) returning duplicate indexes for no reason, as_index=False doesnt help\n
        run this after to clean up the sloppy groupby operation'''
        if df.index.names[0] in df.index.names[1:]:
            df=df.droplevel(0)
        return df
    def consolByCol(df,grupby,consolBy,how='max',sortt=True):
        '''ex: find each row that contains the max of one col (consolBy) in a df\n
        for each unique value in grupby column/index\n
        how: can be str of func to feed .transform(how)\n
        Optional sortt will .sort_values(consolBy)\n
        NOT in place\n
        returns consolidated df'''
        idx = df[consolBy] == df.groupby(grupby)[consolBy].transform(how)
        outdf = df[idx]
        if sortt:
            outdf = outdf.sort_values(consolBy)
        return outdf
    def npfromPy2(npyFile,toDF=True):
        outy = np.load(npyFile,encoding = 'latin1',allow_pickle=True)
        if toDF:
            outy = pd.DataFrame(outy)
        return outy
    def datarray(data,cols=None):
        '''works well for HDF files
        pd.DataFrame(np.array(data))'''
        return pd.DataFrame(np.array(data),columns=cols)
    def unByte(df):
        '''Are b'strings' byting you in the ass? This will get rid of them'''
        #TODO test that this doesn't lose cols
        #return df.applymap(lambda x: x.decode('utf-8'))
        # debyte = lambda x: x.decode('utf-8')
        # return df.apply(lambda s: s.map(debyte))
        str_df = df.select_dtypes(['object']) #np.object may be the older way
        str_df = str_df.stack().str.decode('utf-8').unstack()
        for col in str_df:
            df[col] = str_df[col]
        return df

    def readhdf(hdf,fillvalue=-9999):
        '''hdf: h5py HDF table\n
        returns as pd df\n
        removes pesky b'strings' when applicable'''
        df = xpd.datarray(hdf)
        # if 'object' in res.dtypes.to_list(): #b'str' 
        bcols = [ col for col in df.columns if isinstance(df[col].iloc[0], (bytes, bytearray)) ]
        testbcols = [ col for col in df.columns if isinstance(df[col].iloc[-1], (bytes, bytearray)) ]
        assert bcols==testbcols
        try:
            df[bcols] = df[bcols].applymap(lambda x: x.decode('utf-8'))
        except Exception as e:
            print(f'WARNING: {e} while unbyting {hdf}, resuming')
        df = df.replace({fillvalue:np.NaN})
        return df
    def tohdf(DF,HDFparentGroup,HDFdsName,attrz=None,fillvalue=-9999):
        '''because pd.to_hdf() SUCKS!\n
        HDFparentGroup: h5py Group\n
        HDFgroupName: str, name of new table to add in HDf parent group\n
        returns HDFparentGroup[HDFgroupName]'''
        df = DF.copy()

        if df.columns.dtype=='O':
            df.columns=df.columns.str.replace('/','-') #TODO more illegal chars? replcaemulti
        
        df = df.fillna(fillvalue)
        
        #encode str to bytes
        bcols = [ col for col in df.columns if isinstance(df[col].iloc[0], str) ]
        testbcols = [ col for col in df.columns if isinstance(df[col].iloc[-1], str) ]
        assert bcols==testbcols
        df[bcols] = df[bcols].applymap(lambda x: x.encode('utf-8')).astype('bytes')

        if isinstance(df.columns,pd.RangeIndex): #no col names
            arr = df.values
        else:
            # extract column names and dtypes to create the recarray dtype
            arr_dt = []   
            for col in df.columns:
                arr_dt.append( (col, df[col].dtype) )   

            # create an empty recarray based on Pandas dataframe row count and dtype
            arr = np.empty( (len(df),), dtype=arr_dt )

            # load dataframe column values into the recarray fields
            for col in df.columns:
                arr[col] = df[col].values

        # write to file
        if HDFdsName in HDFparentGroup.keys():
            del HDFparentGroup[HDFdsName]
        HDFparentGroup.create_dataset(HDFdsName,data=arr)
            # throws err if ds is not the same size:
            # HDFparentGroup[HDFdsName][:] = arr

        tbl = HDFparentGroup[HDFdsName]
        if attrz:
            tbl.attrs.update(attrz)
        return tbl
    def tohdfFile(DF,HDFfile,pathToParentGroup,HDFdsName):
        '''tohdf but with unopen! HDF file Path specified\n
        xpd.tohdfFile(df,hdfFile,"/Geometry/Land Cover (Manning's n)",'Calibration Table')
        '''
        with h5py.File(HDFfile,'r+') as hdf:
            xpd.tohdf(DF,hdf[pathToParentGroup],HDFdsName)

    def readhdfAttrs(hdfattrs):
        '''hdfattrs: h5py.File(HDFpth)['table'].attrs'''
        Attrs = pd.Series(hdfattrs)
        Attrs = Attrs.map(lambda b:b.decode('utf-8') if isinstance(b,np.bytes_) else b)
        return Attrs
    def drop_y(df,suff='_y'):
        '''in place- list comprehension of the cols that end with '_y'''
        to_drop = [x for x in df if x.endswith(suff)]
        df.drop(to_drop, axis=1, inplace=True)
    def dropDups(df,axis='cols',inplace=False):
        '''remove duplicate columns from a dataframe\n
        
        How it works:

        Suppose the columns of the data frame are ['alpha','beta','alpha']

        df.columns.duplicated() returns a boolean array: a True or False for each column. If it is False then the column name is unique up to that point, if it is True then the column name is duplicated earlier. For example, using the given example, the returned value would be [False,False,True].

        Pandas allows one to index using boolean values whereby it selects only the True values. Since we want to keep the unduplicated columns, we need the above boolean array to be flipped (ie [True, True, False] = ~[False,False,True])

        Finally, df.loc[:,[True,True,False]] selects only the non-duplicated columns using the aforementioned indexing capability.

        Note: the above only checks columns names, not column values.
        TODO drop rows
        '''
        if axis=='cols' or axis==1:
            res = df.loc[:,~df.columns.duplicated()]
        else:
            assert False, 'TODO drop rows'
        if inplace:
            df = res
            assert False, 'Untested case'
        else:
            return res
    def plotPctNotNull(df):
        '''
        bar chart percent of columns with data
        '''
        chem_var_na = df.isna().sum(axis=0).T

        fig, ax = plt.subplots()
        # modify the default figure size
        fig.set_size_inches(10,40)
        # create the ticks for a horizontal bar plot
        ticks = np.arange(len(chem_var_na))
        # bar plot of percent not NaN
        ax.barh(ticks,
            100-100*(chem_var_na/float(nsample)))
        # set labels for y-axis
        ax.set_yticks(ticks)
        ax.set_yticklabels(chem_var_na.index,fontdict={'fontsize':10})
        # adjust y-axis limits
        ax.set_ylim(-1,len(chem.columns))
        # invert y-axis so meta-data is at the bottom
        ax.invert_yaxis
        # change x-axis limits
        ax.set_xlim(0,100)
        # set title
        ax.set_title('GW chemistry SA:\n{} samples'.format(nsample))
        # set x axis label
        ax.set_xlabel('Percent samples with data')
        # add grid
        return ax.grid()

    def consol(df,grupby=None,pivotCols=None,lvl=None,ND=False,how='list'):
        '''pd prefix-consolidate rows, merge dup values together as lists (or specified how='function')\n
        can pass either a col index, a list of col 'names', etc to grupby (i think)\n
        (pivotCols,Val do nothing in this case)\n
        ND = N-Dimensional - use pivot table instead, (lvl does nothing here)\n
        ex:\n
        -make a DF with columns Weir STA and Z as lists, grouping by and preserving multi-index:\n
        consol(DF,DF.index.names)[['Weir STA','Z']]\n'''
        if how == 'list':
            aggfunk = lambda x: tuple(x) #list can have errors while tuple is better? Can't remember
        else:
            aggfunk = how

        if ND:
            df=df.pivot_table(grupby,pivotCols,aggfunc=aggfunk)
        else:
            df=df.groupby(grupby,level=lvl)
            df=df.agg(aggfunk).applymap(list)#.reset_index() TEST commenting this may mess up other functions depending on xpd.consol!!!
        #if how == 'list':
            #df=df.agg(lambda x: tuple(x)).applymap(list).reset_index() #why reset index?
        #else:
        return df
    def explode(df, lst_cols, fill_value='', preserve_index=True):
        '''returns DF of ---splits each csv field and create a new row per entry\n
        make sure `lst_cols` is list-alike \n
        exs:\n
        explode(df, ['num','text']\n
        explode(df.assign(var1=df.var1.str.split(',')), 'var1')\n
        using this little trick we can convert CSV-like column to list column:\n
        df.assign(var1=df.var1.str.split(','))
        TODO is this different than\n
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.explode.html
        '''

        #https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows/40449726#40449726
        
        if (lst_cols is not None
            and len(lst_cols) > 0
            and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
            lst_cols = [lst_cols]
        # all columns except `lst_cols`
        idx_cols = df.columns.difference(lst_cols)
        # calculate lengths of lists
        lens = df[lst_cols[0]].str.len()
        # preserve original index values    
        idx = np.repeat(df.index.values, lens)
        # create "exploded" DF
        res = (pd.DataFrame({
                    col:np.repeat(df[col].values, lens)
                    for col in idx_cols},
                    index=idx)
                .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                                for col in lst_cols}))
        # append those rows that have empty lists
        if (lens == 0).any():
            # at least one list in cells is empty
            res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                    .fillna(fill_value))
        # revert the original index order
        res = res.sort_index()
        # reset index if requested
        if not preserve_index:        
            res = res.reset_index(drop=True)
        return res
    def sort5yrMulti(df,ascend=False,dex=['Flow','Name']):
        '''hardcoded for 2 lvl index, sorts both levels .sort_index(ascending=ascend)
        hardcoded for the format "5YR"'''
        df=df.reset_index()
        df[dex[0]]=df[dex[0]].str.replace('YR','').astype(int) #'5YR' -> 5
        df=df.set_index(dex).sort_index(ascending=ascend)
        df.index=df.index.map(lambda x: ((str(x[0])+'YR'),x[1]))
        return df
    def sort5yr(df,lvl=0,ascend=False,inplace=False):
        '''sorts by integer extracted from index level lvl\n
        '''
        res = df.iloc[df.index.get_level_values(lvl).str.extract('(\d+)', expand=False).astype(int).argsort()]
        # print(res)
        if not ascend:
            res=res.iloc[::-1]
        if inplace:
            assert False,'inplace dont work!'
            df = res
            # print('SORRRRRTED',df)
        else:
            return res
    def multiMerge(df1,df2,dex=0, drop_dups=True,how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False, suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
        '''Correctly handles merge w/ multiindexed columns\n
        dex will set the index of the merged df to {0:just reset, 1:df1,2:df2}\n
        reset_index will reset indices of df1 & df2 (for the merge, not inplace)\n
        TODO multiindex index'''
        # if reset_index:
        dfL,dfR=df1.reset_index(),df2.reset_index()
        # else:
        # dfL,dfR=df1,df2
        q=dfL.merge(dfR,how,on,left_on,right_on,left_index,right_index, sort, suffixes, copy, indicator, validate)
        cols = q.columns.to_list()
        cols = [col if isinstance(col,tuple) else (col,'') for col in cols ]
        q.columns = pd.MultiIndex.from_tuples(cols)
        if drop_dups:
            q=xpd.dropDups(q)
        dexs = {1:df1.index.names,2:df2.index.names}
        if dex:
            q=q.set_index(dexs[dex]).sort_index()
        return q
    def findNeighbors(df,field,value):
        ''''''
        exactmatch=df[df[field]==value]
        if not exactmatch.empty:
            return exactmatch.index
        else:
            lowerneighbour_ind = df[df[field]<value][field].idxmax()
            upperneighbour_ind = df[df[field]>value][field].idxmin()
            return [lowerneighbour_ind, upperneighbour_ind]
    def read_excel_sheets(xls_path):
        """Read all sheets of an Excel workbook and return a single DataFrame\n
        can be as .xls, .xlsx, etc
        https://gist.githubusercontent.com/NicoleJaneway/64103ea4a2e89ad19acdf1e68e633e02/raw/c0ce333c845a246242e33388c8891ed8d9084f92/sheet_name.py"""
        print(f'Loading {xls_path} into pandas')
        xl = pd.ExcelFile(xls_path)
        df = pd.DataFrame()
        columns = None
        for idx, name in enumerate(xl.sheet_names):
            print(f'Reading sheet #{idx}: {name}')
            sheet = xl.parse(name)
            if idx == 0:
                # Save column names from the first sheet to match for append
                columns = sheet.columns
            sheet.columns = columns
            # Add sheet name as column
            sheet['sheet'] = name
            # Assume index of existing data frame when appended
            df = df.append(sheet, ignore_index=True)
        return df
    
try:
    import xarray as xr
except Exception as e:
    if printImportWarnings:
        print('WARNING: ',e)
try:
    import zarr
except Exception as e:
    if printImportWarnings:
        print('WARNING: ',e,'Zarr is an optional dependency')
class xxr:
    def RMSE(y_true,y_sim,dim=None,normed=True):
        ''' ## Root Mean Squared Error\n
        \mathrm{RMSD} = \sqrt{\frac{\sum_{i=1}^{N}\left(x_{i}-\hat{x}_{i}\right)^{2}}{N}}\n\n
        - y_true	=	actual observations time series
        - y_sim	=	estimated time series\n
        - N	=	number of non-missing data points
        y_true,y_sim: xr da's or np arrays of true/observed values vs simulated/predicted\n
        normed: whether to normalize to percentage of y_true:\n
        RMSE*( N / y_sim.sum() ) \n\n
        dim='t', named xr da dim to aggregate along
        '''
        dimm = {'dim':dim} if dim else {}
        rms = np.sqrt(np.square( y_sim - y_true ).mean(**dimm))
        if normed:
            # rms = rms*y_true.count(**dimm)/y_true.sum(**dimm)
            rms = rms/np.abs(y_true.mean(**dimm))
            
        return rms
    def pctBias(y_true,y_sim,dim=None,asPct=False):
        ''' ## Percent Bias
        - y_true	=	actual observations time series
        - y_sim	=	estimated time series\n
        y_true,y_sim: xr da's or np arrays of true/observed values vs simulated/predicted\n
        dim='t', named xr da dim to aggregate along\n
        returns as ratio unless aspct (multiply by 100 to get percent)\n
        '''
        dimm = {'dim':dim} if dim else {}
        pct = 100 if asPct else 1
        pb = pct*(y_sim.mean(**dimm) - y_true.mean(**dimm)) / y_true.mean(**dimm)
        return pb

    def errMetricTrio(y_true,y_sim,dim=None,
        # RMSE_normed=True,pctBias_asPct=False
        ):
        '''cor,rmse,pb = [ funk(**kwargz)  for funk in (xr.corr,xxr.RMSE,xxr.pctBias) ]'''
        kwargz = {'dim':dim} if dim else {}
        # corr = xr.corr if dim else np.correlate
        # metrix = [ funk(y_true,y_sim,**kwargz)  for funk in (corr,xxr.RMSE,xxr.pctBias) ]
        metrix = [ funk(y_true,y_sim,**kwargz)  for funk in (xr.corr,xxr.RMSE,xxr.pctBias) ]
        trio = xr.Dataset(dict(zip(['cor','rmse','pb'],metrix)))
        return trio
    def to_file(ds,outFile,complevel=5,driver='inferFromExtension'):
        '''
        bounce to .nc or .zarr at compression level complevel
        '''
        if outFile.name.split('.')[-1] == 'nc' or driver.lower() in {'netcdf','netcdf4','nc'}:
            print('Writing to NetCDF4')
            compr = dict(zlib=True, complevel=complevel)
            encoding = encoding = {ds.name: compr} if isinstance(ds,xr.DataArray) \
                else {var: compr for var in ds.data_vars}
            ds.to_netcdf(outFile, encoding=encoding)
            print(f'Successfully bounced to {outFile}')
        
        elif outFile.name.split('.')[-1] == 'zarr' or driver.lower()=='zarr':
            print('Writing to Zarr')
            compr = {'compressor':
                zarr.Blosc(cname="zstd", clevel=7, shuffle=2) }

            encoding = encoding = {ds.name: compr} if isinstance(ds,xr.DataArray) \
                else {var: compr for var in ds.data_vars}
            ds.to_zarr(outFile, encoding=encoding)
            print(f'Successfully bounced to {outFile}')