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
try:
    from tqdm import tqdm
except Exception as e:
    print(f'WARNING: {e}')

regxASCII = "[^\W\d_]+"
plotlystrftime = '%Y-%m-%d %X' # for plotly timestamps

def GB(b):
    return b*9.3132257461548E-10 

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
def progbar(func):
    '''decorator\n
    prints prog(iprog,itot) when iprog,itot added to func kwargs'''
    def wrapper(*args,iprog=None,itot=None, **kwargs):
        # before
        if iprog:
            prog(iprog,itot)

        ret = func(*args, **kwargs)
        return ret
    return wrapper
import contextlib

def benchmark(funk,argz):
    '''
    Benchmark \n
    funk: function to benchmark \n
    argz=[(),()]
    '''
    argzs = argz[:]
    if not isiter(argzs[0]):
        argzs = [ (argz,) for argz in argzs]

    def bmark(argz):
        print(f'Args: {argz}')

        start = datetime.now()
        res = funk(*argz)
        end = datetime.now()

        t = (end - start).total_seconds()
        print(f'Time: {t} s\n')
        return t
    times = [bmark(argz) for argz in tqdm(argzs) ]
    print(f'{argzs[ np.argmin(times) ]}\nis the fastest')
    return times
    

red = '\033[91m' 

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
    ARRGHHHSSSS = [str(arg) for arg in ARRGHHHSSSS]
    p = subprocess.Popen([cmd,*ARRGHHHSSSS],
         stdout=subprocess.PIPE,stderr=subprocess.PIPE, bufsize=1)
    for line in iter(p.stdout.readline, b''):
        print(line.split(b'>')[-1]),
    p.stdout.close()
    for line in iter(p.stderr.readline, b''):
        print(f"{red}{line.split(b'>')[-1]}"),
    p.wait()

    elaps = datetime.now() - thenn
    return elaps

try:
    import wmi
    import psutil
    import socket

    def getCPU(remote_ip_address, username, password):
        '''returns cpu usage in % (int) on remote ip'''
        local_ip_address = socket.gethostbyname(socket.gethostname())

        remote = {} if socket.gethostbyname(remote_ip_address) == local_ip_address else \
             dict(computer=remote_ip_address, user=username, password=password)
        c = wmi.WMI(find_classes=False, **remote )

        return sum(
            [ i.LoadPercentage
                for i in c.Win32_Processor() if i.LoadPercentage ] )
            # print("%s %s" % (i.DeviceID, i.LoadPercentage))
except Exception as e:
    if printImportWarnings:
        print('WARNING: ',e)

from datetime import datetime
def lstat(file):
    '''returns last access time of file'''
    return pd.Timestamp(datetime.fromtimestamp(file.lstat().st_mtime))
def getCommonPath(files):
    common_path = Path(files[0])
    for file in files[1:]:
        while not str(file).startswith(str(common_path)):
            common_path = common_path.parent
    return common_path
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
    return pthBackup/filee.name
def updateLstat(f):
    tmp = f.with_suffix(f'.tmp{f.suffix}')
    shutil.copy(f,tmp)
    return shutil.move(tmp,f)
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
    '''recursive, pth: Path or str\n
    in GB'''
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(str(pth)):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return GB(total_size)
    
def nan2None(obj):
    if isinstance(obj, dict):
        return {k:nan2None(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [nan2None(v) for v in obj]
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    return obj
class NanConverter(json.JSONEncoder):
    def default(self, obj):
        # possible other customizations here 
        pass
    def encode(self, obj, *args, **kwargs):
        obj = nan2None(obj)
        return super().encode(obj, *args, **kwargs)
    def iterencode(self, obj, *args, **kwargs):
        obj = nan2None(obj)
        return super().iterencode(obj, *args, **kwargs)
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
        if isinstance(htmlpth,str):
            if not Path(htmlpth).exists():
                return '',htmlpth # if it's just a str of html body TODO better way
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

def URLfromParams(baseURL,params):
    '''returns URL with params for API calls etc given\n
    params: urlparams as dict\n
    baseURL: str'''
    return baseURL + '?' + \
        '&'.join(
            [f'{key}={val}' for key,val in params.items()]
        )
def DL(url,toPth,filename='infer',callback=None):
    '''DL's file from url and places in toPth Path\n
    returns request status code'''
    if not callback:
        print("downloading: ",url)
    if not toPth.exists():
        toPth.mkdir()
    # assumes that the last segment after the / represents the file name
    # if url is abc/xyz/file.txt, the file name will be file.txt
    if filename=='infer':
        file_name_start_pos = url.rfind("/") + 1
        file_name = url[file_name_start_pos:]
    else:
        file_name=filename

    r = requests.get(url, stream=True, 
        verify=False
    )
    if r.status_code == requests.codes.ok:
        with open(toPth/file_name, 'wb') as f:
            for data in r:
                f.write(data)
    else:
        print(f'''WARNING for {url}
{r.status_code}: {r.reason}''')
    if callback:
        callback()
    return r.status_code

from multiprocessing.pool import ThreadPool

@timeit
def DLmulti(urls,toPth,filenames='infer',threds=12):
    '''DL multi async\n
    will infer filenames if filenames=='infer', otherwise
    must specify a list of len(urls) of file names with suffix, to download to toPth
    '''
    toPth.mkdir(parents=True,exist_ok=True)
    #TODO starmap_async?
    fnames = ['infer']*len(urls) if filenames=='infer' else filenames
    results = ThreadPool(threds).starmap(DL, 
        [ 
            ( url,toPth,fname,
                prog(i,len(urls),f'Downloading {url}') ) 
            for i,(url,fname) in enumerate(zip(urls,fnames)) 
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

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Coroutine, TypeVar

def async_to_sync(coroutine, timeout: float = 30) -> TypeVar("T"):
    '''cant run twice???'''
    T = TypeVar("T")
    def run_in_new_loop():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    if threading.current_thread() is threading.main_thread():
        if not loop.is_running():
            return loop.run_until_complete(coroutine)
        else:
            with ThreadPoolExecutor() as pool:
                future = pool.submit(run_in_new_loop)
                return future.result(timeout=timeout)
    else:
        return asyncio.run_coroutine_threadsafe(coroutine, loop).result()

import zipfile
import gzip

def gunzip(gzfile, outfile):
    # Ensure input and output paths are treated as Path objects for consistency
    infiletemp = Path(gzfile)
    templocsys = Path(outfile)
    
    # Use gzip library to decompress the file
    with gzip.open(infiletemp, 'rb') as f_in:
        with open(templocsys, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

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

def gravitate(f):
    if f.is_dir():
        return [gravitate(i) for i in f.iterdir()]
    ftmp = f.parent/f'{f.name}_temp'
    shutil.copy(f,ftmp)
    f.unlink()
    shutil.move(ftmp,f)
    return f

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
def attrsFromDict(obj,cfgdict):
    [setattr(obj,key,val) for key,val in cfgdict.items()]

def bold(st):
    '''for plots\n
    wrap st with <b> tags, apply to each st in iterable if st is iterable'''
    if isiter(st):
        return [bold(i) for i in st]
    return st if str(st).startswith('<b>') else f'<b>{st}</b>'
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

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
import json
from collections import UserDict

class filedict(UserDict):
    def __init__(self, json_file):
        '''init with jsonfile Path\n
        keeps jsonfile updated with itself\n
        inits with jsonfile if exists, to allow state to persist over multiple script calls, like a DB
        '''
        self.json_file = json_file
        # Init with default dict if jsonfile doesn't exist
        self.data = self.read()

    def read(self):
        if self.json_file.exists():
            with open(self.json_file,'r') as f:
                return json.load(f)
        else:
            return {}

    def write(self):
        self.json_file.parent.mkdir(parents=True,exist_ok=True)
        with open(self.json_file, 'w') as outfile:
            json.dump(self.data, outfile,indent=4)

    def __setitem__(self,key,val):
        super().__setitem__(key,val)
        self.write()

    def __delitem__(self,key):
        super().__delitem__(key)
        self.write()

    def update(self, items=(), **kwds):
        super().update(items, **kwds)
        self.write()

    def pop(self, key, d=None):
        result = super().pop(key, d)
        self.write()
        return result

    def popitem(self):
        result = super().popitem()
        self.write()
        return result

    def clear(self):
        super().clear()
        self.write()


# class filedict(dict):
#     def __init__(self,jsonfile):
#         self.jsonfile = jsonfile
#         super().__init__(self.read())
#     '''init with jsonfile Path\n
#     keeps jsonfile updated with itself\n
#     inits with jsonfile if exists, to allow state to persist over multiple script calls, like a DB
#     '''
#     def read(self):
#         if self.jsonfile.exists():
#             if len(self.jsonfile.read_text()):
#                 return json.load(open(self.jsonfile,'r'))
#             else:
#                 return {}
#         else:
#             return {}
#     def write(self):
#         self.jsonfile.parent.mkdir(parents=True,exist_ok=True)
#         with open(self.jsonfile, 'w') as outfile:
#             json.dump(self, outfile,indent=4)
#     def __delitem__(self,key):
#         super().__delitem__(key)
#         self.write()
#     def __setitem__(self,key,val):
#         super().__setitem__(key,val)
#         self.write()
#     def update(self,items):
#         super().update(items)
#         self.write()
#     def pop(self,item):
#         super().pop(item)
#         self.write()
        
def dictSwap(aDict):
    return {value:key for key, value in aDict.items()}
class flexList(list):    
    def __missing__(self, key):
        print('missing!')
        flexDict({})
    def __getitem__(self, __key):
        try:
            res = super().__getitem__(__key)
        except IndexError:
            return fslexDict({})
        if isinstance(res,list):
            return flexList(res)
        if isinstance(res,dict):
            return flexDict(res)
# ax='x'
# ts['layout'][f'{ax}axis']['type']
# kwargz={}
# lyt = ts['layout']
# for ax in ('x','y'):
#     axlvl = lyt[f'{ax}axis']
#     if axlvl['type']=='log':
#         kwargz[f'log{ax}']=True
#     if axlvl['title']['text']:
#         kwargz[f'{ax}label'] = axlvl['title']['text']
# kwargz
# hv.Curve({'x':ts['x'],**ts['data'][0]}).opts(**kwargz)
        
class flexDict(dict):    
    def __missing__(self, key):
        return flexDict({})
    def __getitem__(self, __key):
        res = super().__getitem__(__key)
        if isinstance(res,list):
            return flexList(res)
        if isinstance(res,dict):
            return flexDict(res)
        return res

import re
def replaceMulti(adict, text):
    '''Note: happens in one pass!\n
    returns the new str'''
    # Create a regular expression from all of the dictionary keys
    regex = re.compile("|".join(map(re.escape, adict.keys(  ))))
    # For each match, look up the corresponding value in the dictionary
    return regex.sub(lambda match: adict[match.group(0)], text)
def wordCount(text):
    return len(re.findall(r'\w+', text))
def ondah(namee):
    '''legalizes names for use in a SQL DB by adding a million underscores'''
    nm = namee[:]
    unsavoryCharacters = [' ','.','`','=','+','^',',',r'/','\\','?',"'"]
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
    def topNmean(aray,n):
        p = aray
        p = p.flatten()
        p = -np.sort(-p)
        ptop = p[:n]
        topq = np.mean(ptop)
        return topq
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
        snp.loc[snp[0]==snp.index,0]=np.nan
        snp.loc[snp[1]==snp.index,1]=np.nan
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
    def quantiles(values,probabilities,quantiles = np.linspace(0, 1, 101)):
        """
        Compute the approximate inverse cumulative distribution function (CDF) or quantile function.\n
        \n
        Parameters:\n
            values (list or array-like): Values of the probability mass function.\n
            probabilities (list or array-like): Corresponding probabilities for each value.\n
            quantiles (list or array-like, optional): Quantiles to compute the inverse CDF values.\n
                Default is np.linspace(0, 1, 100).\n
        \n
        Returns:\n
            list: Inverse CDF values corresponding to the provided quantiles.\n
        \n
        Example:\n
            values = [1, 2, 3, 4, 5]\n
            probabilities = [0.1, 0.2, 0.3, 0.2, 0.2]\n
            quantiles = np.linspace(0, 1, 101)\n
            result = quantiles(values, probabilities, quantiles)\n
        """
        # Assuming you have two columns: 'values' and 'probabilities'

        # Sort values and probabilities in ascending order based on values
        sorted_data = sorted(zip(values, probabilities), key=lambda x: x[0])
        values, probabilities = zip(*sorted_data)

        # Compute the cumulative probabilities
        cumulative_probs = np.cumsum(probabilities)

        # Define the inverse CDF function
        inverse_cdf = np.interp

        # Compute the inverse CDF values
        inverse_cdf_values = inverse_cdf(quantiles, cumulative_probs, values)
        return inverse_cdf_values
from scipy.interpolate import interp1d

try:
    import asyncio, aiofiles
    from io import StringIO
except Exception as e:
    print('WARNING: ',e)

try:
    import h5py
except Exception as e:
    print('WARNING: ',e)
import numpy as np
if __package__ is None or __package__ == '':
    # uses current directory visibility
    from byteme import set_attrs_bytes,h5_nullterm_dtype
else:
    # uses current package visibility
    from .byteme import set_attrs_bytes,h5_nullterm_dtype

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
    
    def hasna(df):
        if np.isnan(df) is True:
            return True
        try:
            na = df.isna().any()
            if isinstance(na,pd.Series):
                return na.any()
            else:
                return na
        except AttributeError as e:
            return np.isnan(df)

    hasnan = hasna
            
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
    def nearest(val,lookupSeries):
        '''Find the nearest index value in lkup series to val'''
        lkup = lookupSeries
        idx = lkup.sub(val).abs().idxmin()
        return idx
    def fillBetween(series):
        """
        Fill NaN values in a Pandas Series only between valid numbers, leaving 
        leading and trailing NaN values untouched.

        This function interpolates missing values within the valid range of 
        non-NaN values while preserving NaNs at the start and end of the Series.

        Parameters
        ----------
        series : pd.Series
            The input Pandas Series containing NaN values and valid numbers.

        Returns
        -------
        pd.Series
            A Series with NaN values filled only within the valid range of numbers.

        Examples
        --------
        df\n
            +-----+-----+-----+
            |  t  |  a  |  b  |
            +-----+-----+-----+
            |  0  | NaN | NaN |
            |  1  | 2.0 | NaN |
            |  2  | NaN | 4.0 |
            |  3  | 2.0 | NaN |
            |  4  | NaN | 4.0 |
            |  5  | NaN | NaN |
            +-----+-----+-----+

        df = df.apply(xpd.fillBetween, axis=0)\n

            +-----+-----+-----+
            |  t  |  a  |  b  |
            +-----+-----+-----+
            |  0  | NaN | NaN |
            |  1  | 2.0 | NaN |
            |  2  | 2.0 | 4.0 |
            |  3  | 2.0 | 4.0 |
            |  4  | NaN | 4.0 |
            |  5  | NaN | NaN |
            +-----+-----+-----+
        """
        mask = series.notna()
        valid_range = (mask.cumsum() > 0) & (mask[::-1].cumsum()[::-1] > 0)
        filled = series.interpolate()
        return filled.where(valid_range)

        mask = series.notna()
        # Boolean mask for valid range
        valid_range = (mask.cumsum() > 0) & (mask[::-1].cumsum()[::-1] > 0)
        # Interpolate the series and mask out invalid ranges
        filled = series.interpolate()
        return filled.where(valid_range)

    def roundWithoutExceeding(series,maxsum='original_sum'):
        """
        Rounds elements of a Pandas series to the nearest integer 
        in such a way that the sum of the rounded series does not exceed the sum 
        of the original series.

        Parameters:
        series (pd.Series): Series of numbers to be rounded.

        Returns:
        pd.Series: A new series where all numbers are rounded to the nearest integer, 
        but the sum does not exceed the original sum.
        """
        if maxsum=='original_sum':
            original_sum = series.sum()
        else:
            original_sum = maxsum
        rounded = series.round()
        difference = rounded.sum() - original_sum

        # if the sum of the rounded series is greater than the original sum
        if difference > 0:
            # get indices of elements where the fractional part is less than 0.5
            indices = series[series % 1 < 0.5].index
            # decrement these elements until the sum of the rounded series is not more than the original sum
            while difference > 0 and len(indices) > 0:
                for i in indices:
                    rounded[i] -= 1
                    difference = rounded.sum() - original_sum
                    if difference <= 0:
                        break
                indices = rounded[rounded < series].index

        return rounded

    async def _read_csv_async(file_path):
        async with aiofiles.open(file_path, mode='r') as f:
            content = await f.read()
            return content
    async def read_csv_async(csvpth,**kwargz):
        content = await xpd._read_csv_async(csvpth)
        data = StringIO(content)
        df = pd.read_csv(data,**kwargz)
        return df
    async def to_csv_async(df, file_name, **kwargs):
        '''This function coverts the DataFrame into CSV format using `DataFrame.to_csv` and saves it as a string. Later, this string is written to a file using the `aiofiles.open`. '''
        csv_data = df.to_csv(**kwargs)
        async with aiofiles.open(file_name, mode='w') as file:
            await file.write(csv_data)

    def tsGaps(dtIndex_ts,freq='H'):
        '''index should be pd datetime index!
            #The complete timeseries record needed\n
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
        #return df.map(lambda x: x.decode('utf-8'))
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
            df[bcols] = df[bcols].map(lambda x: x.decode('utf-8'))
        except Exception as e:
            print(f'WARNING: {e} while unbyting {hdf}, resuming')
        df = df.replace({fillvalue:np.nan})
        return df
    def forceDtypes(DF,dtypeDict={int:np.int8,float:np.float32}):
        '''forces dtypes in DF according to dtypeDict\n
        dtypeDict keys are python types, values are numpy dtypes\n
        ex: xpd.forceDtypes(df,{int:np.int8,float:np.float32})'''
        df = DF.copy()
        for pytype, nptype in dtypeDict.items():
            cols = df.select_dtypes(include=[pytype]).columns
            df[cols] = df[cols].astype(nptype)
        return df

    def _clone_dataset_kwargs(ds):
        """Extract dtype + layout info from an existing h5py dataset."""
        return dict(
            dtype=ds.dtype,
            chunks=ds.chunks,
            compression=ds.compression,
            compression_opts=ds.compression_opts,
            shuffle=ds.shuffle,
            fletcher32=ds.fletcher32,
            fillvalue=ds.fillvalue,
        )

    def tohdf(DF, HDFparentGroup, HDFtblName, attrz=None, fillvalue=None, like=None,
              force_nullterm=True, **kwargs):
        """
        Write DataFrame to HDF5 dataset.
        
        like: h5py.Dataset to clone schema from (auto-uses existing if present)
        kwargs: compression, chunks, etc.
        fillvalue: value to fill NaNs with (sometimes need -9999)
        attrz: dict of attributes to add to dataset
        force_nullterm: force null-terminated strings (h5py default is not null-terminated)
        Returns the created h5py.Dataset.
        """
        df = DF.copy()
        # Auto-use existing as template (only if compound)
        if HDFtblName in HDFparentGroup.keys():
            like = HDFparentGroup[HDFtblName] if like is None else like
            del HDFparentGroup[HDFtblName]
        print(HDFtblName)
        print(like)

        
        if df.columns.dtype==str:
            df.columns=df.columns.str.replace('/','-') #TODO more illegal chars? replcaemulti
        if fillvalue is not None:
            df = df.fillna(fillvalue)

        #encode str to bytes
        bcols = [ col for col in df.columns if isinstance(df[col].iloc[0], str) ]
        testbcols = [ col for col in df.columns if isinstance(df[col].iloc[-1], str) ]
        assert bcols==testbcols
        df[bcols] = df[bcols].map(lambda x: x.encode('utf-8')).astype('bytes')

        if  like:
            is_compound = like.dtype.names is not None
            is_matrix   = (not is_compound) and like.ndim == 2
        else:
            is_matrix = (
                df.ndim == 2 and
                df.dtypes.nunique() == 1 and
                df.dtypes.iloc[0].kind in "iufb"   # int/uint/float/bool
            )
        # print(f'is_matrix: {is_matrix}')
        if is_matrix:
            arr = df.to_numpy(dtype=like.dtype if like else None, copy=False)
        else:
            arr = df.to_records(index=False)

        if like:
            create_kwargs = {
                'compression': like.compression,
                'compression_opts': like.compression_opts,
                'dtype': like.dtype,# if is_matrix else (like.dtype,),
                'maxshape':like.maxshape,
                'shuffle': like.shuffle,
                'fillvalue': like.fillvalue,
                **kwargs
            } 

            # Clone compression/chunk settings from like, adjusting chunks for 1D shape (recarray)
            c = getattr(like, "chunks", None)
            chunks = c if is_matrix else (c[0],)
            chunks = None if c is None else chunks
            if chunks:
                create_kwargs['chunks'] = chunks
        else:
            create_kwargs = {
                'dtype': arr.dtype ,
                'maxshape': (None,) + arr.shape[1:] if is_matrix else (None,),
                **kwargs
            }
        print(create_kwargs)
        
        if force_nullterm:
            # force null-terminated strings
            create_kwargs['dtype'] = h5_nullterm_dtype(create_kwargs['dtype'])
        dset = HDFparentGroup.create_dataset(HDFtblName, 
                                         shape=arr.shape,
                                      **create_kwargs)
        dset[:] = arr
        
        if attrz:
            # battrz = { k:(v.encode('utf-8') if isinstance(v,str) else v) for k,v in attrz.items() }
            # dset.attrs.update(battrz)

            set_attrs_bytes(dset, attrz)

        
        return dset


            #TODO use this instead
            #             modify(name, value)
            # Change the value of an attribute while preserving its type and shape. Unlike AttributeManager.__setitem__(), if the attribute already exists, only its value will be changed. This can be useful for interacting with externally generated files, where the type and shape must not be altered.

            # If the attribute doesn't exist, it will be created with a default shape and type.

            # Parameters
            # :
            # name (String) Name of attribute to modify.

            # value New value. Will be put through numpy.array(value).
            # for atr in ['Percent Impervious Filename','Infiltration Filename','Land Cover Filename']:
            #     hdf['Geometry'].attrs.modify(atr,
            #         hdf['Geometry'].attrs[atr].decode().replace('..\\07_MapLayers','.\\MapLayers').encode()
            #     )
            # with h5py.File(nexthdf, 'r+') as hdf:
            #     grup = hdf['/Event Conditions/Meteorology/Precipitation']
            #     for atr in ['DSS Filename','DSS Pathname']:
            #         st = grup.attrs[atr].decode()
            #         st = replaceMulti(rep,st)
            #         grup.attrs.modify(atr,st.encode())
        return tbl
    def tohdfFile(DF,HDFfile,pathToParentGroup,HDFtblName,
        attrz=None,fillvalue=-9999,**kwargs):
        '''tohdf but with unopen! HDF file Path specified\n
        xpd.tohdfFile(df,hdfFile,"/Geometry/Land Cover (Manning's n)",'Calibration Table')
        '''
        with h5py.File(HDFfile,'r+') as hdf:
            xpd.tohdf(DF,hdf[pathToParentGroup],HDFtblName,
                attrz=attrz,fillvalue=fillvalue,**kwargs)

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
    quantiles = xnp.quantiles

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
            df=df.agg(aggfunk).map(list)#.reset_index() TEST commenting this may mess up other functions depending on xpd.consol!!!
        #if how == 'list':
            #df=df.agg(lambda x: tuple(x)).map(list).reset_index() #why reset index?
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
    def findNeighbors(df, colname, value):
        """
        Fuzzy lookup\n
        Finds the exact matches for a given value in a specified column of a DataFrame or Series. If no exact matches are found, it returns the closest lower and upper neighbors.\n
        Args:\n
            df (pd.DataFrame or pd.Series): The DataFrame or Series to search within.\n
            colname (str): The column name to search in.\n
            value (int or float): The value to find in the specified column.\n
        Returns:\n
            pd.DataFrame or pd.Series: A DataFrame or Series containing the exact matches if they exist. Otherwise, it returns the closest lower and upper neighbors.\n   
            ie len of df will be 1 or 2, respectively\n
        Raises:\n
            AssertionError: If df is not a DataFrame or Series.\n
        """

        assert (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)), type(df)
        exactmatch = df[df[colname] == value]
        if not exactmatch.empty:
            return df.iloc[exactmatch.index]
        else:
            lower_df = df[df[colname] < value]
            upper_df = df[df[colname] > value]
            lowerneighbour_ind = lower_df[colname].idxmax() if not lower_df.empty else None
            upperneighbour_ind = upper_df[colname].idxmin() if not upper_df.empty else None
            indices = [ind for ind in [lowerneighbour_ind, upperneighbour_ind] if ind is not None]
            return df.iloc[indices]

    def read_excel_sheets(xls_path, sheet_names=None):
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
            df = pd.concat([df, sheet], ignore_index=True)
        return df
    readExcelSheets = read_excel_sheets
    
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
    def buffer(ds, dim, buff, method=None, fill_value=np.nan, **kwargs):
        """
        Expands a specified dimension of an xarray dataset by a given buffer, 
        adding new coordinates at the same regular interval without changing any current indices.

        Parameters
        ----------
        ds : xarray.Dataset
            The input xarray dataset containing the dimension to expand.
        dim : str
            The name of the dimension to expand in the dataset.
        buff : float, int, or pd.Timedelta
            The buffer size to be added to both ends of the specified dimension range.
            Can be a float or int for numerical dimensions or a pd.Timedelta for datetime dimensions.
        method : str, optional
            The method to use for filling the new values (default is None).
        fill_value : float, optional
            The value to fill the new coordinates with (default is NaN).

        Returns
        -------
        xarray.Dataset
            A new xarray dataset with expanded dimension, where new 
            coordinates are filled with the specified fill value.

        Notes
        -----
        ds = buffer_1d(bc, dim='time', buff=pd.Timedelta(days=7*4))
        This function retains the original indices in the specified dimension 
        and only adds new indices to extend the dataset by the given buffer.
        """
        coords = ds[dim]
        step = pd.Timedelta(np.abs(coords[1] - coords[0]).values) if np.issubdtype(coords.dtype, np.datetime64) else np.abs(coords[1] - coords[0])

        # Check if the dimension coordinates are datetime objects
        if np.issubdtype(coords.dtype, np.datetime64):
            if not isinstance(buff, pd.Timedelta):
                raise TypeError("For datetime dimensions, buff should be a pandas.Timedelta.")

            # Generate buffer coordinates for datetime
            lower_buffer = pd.date_range(end=coords.min().values - step, periods=int(buff / step), freq=step)
            upper_buffer = pd.date_range(start=coords.max().values + step, periods=int(buff / step), freq=step)
            
        else:
            # Assume numeric type for other dimensions
            if isinstance(buff, pd.Timedelta):
                raise TypeError("For numeric dimensions, buff should be a float or int.")
            
            # Generate buffer coordinates for numeric values
            lower_buffer = np.arange(coords.min() - buff, coords.min(), step)
            upper_buffer = np.arange(coords.max() + step, coords.max() + buff + step, step)
        
        # Concatenate new coordinates with original ones
        new_coords = np.union1d(coords, np.concatenate([lower_buffer, upper_buffer]))
        
        # Reindex dataset with expanded coordinates and fill new values
        ds_expanded = ds.reindex({dim: new_coords}, method=method, fill_value=fill_value, **kwargs)
        
        return ds_expanded
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
    def to_file(ds,outFile,complevel=3,driver='inferFromExtension'):
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
                zarr.Blosc(cname="zstd", clevel=complevel, shuffle=2) }

            encoding = encoding = {ds.name: compr} if isinstance(ds,xr.DataArray) \
                else {var: compr for var in ds.data_vars}
            ds.to_zarr(outFile, encoding=encoding)
            print(f'Successfully bounced to {outFile}')
    # bounce = xxr.to_file # alias
    def benchmarkComplevels(ds,toPth,lvls=[3,5,7,9]):
        '''
        Benchmark nc xxr.to_file compression levels for a dataset\n
        '''
        def bmark(lvl):
            print(f'Compression level: {lvl}')
            outnc = toPth/f'test{lvl}.nc'

            start = datetime.now()
            xxr.to_file(ds,outnc,complevel=lvl)
            end = datetime.now()

            t = (end - start).total_seconds()
            sz = outnc.stat().st_size/1e9
            print(f'Time: {t} s')
            print(f'Size: {sz} GB\n')
            return t,sz
        bm = [bmark(lvl) for lvl in tqdm(lvls) ]
        return bm

    @progbar
    def getWaveRaster(nc,var,clipper=None,
        outTIF=None,outvec=None,outJSON=None,
        func=lambda ds:ds,
        clipbuff=50,res=90):
        '''
        Rasterizes netCDF dataset (nc)\n
        extracts wave data, applies a function (func) to preprocess the data, \n
        and clips the data to a specified region (clipper). The processed data can be saved as a raster TIFF file, \n
        a vector file, or a JSON file. \n

        Args:
            nc (str): Path to the netCDF file. \n
            clipper (str): Path to the clipper shapefile defining the region of interest. \n
            outTIF (str, optional): Path to output raster TIFF file. \n
            outvec (str, optional): Path to output vector file. \n
            outJSON (str, optional): Path to output JSON file to export flat array of
             unclipped ds[var].values.flatten(). \n
            func (function, optional): Function to apply to the dataset during preprocessing.
             Default is an identity function. \n
            clipbuff (int, optional): Buffer for clipping in miles. Default is 50 miles. \n
            res (int, optional): cellsize resolution for the output raster. Default is 90. \n

        Returns:
            xarray.DataArray: Geocube object representing the processed and clipped data. \n
        '''
        # var = nc.stem.split('_')[1]
        wv = xr.open_dataset(nc,
            drop_variables=['neta','nvel'],
            # chunks={'node':1},
            )
        # wv = wv.rename({f'swan_{var}_max':var})
        
        wv = func(wv)

        if outJSON:
            dst = wv[var].values
            dst = dst[~np.isnan(dst)]
            with open(outJSON, 'w') as outfile:
                json.dump(list(dst), outfile)
            print(f'distribution saved to {outJSON}')
            
        nodes = gpd.GeoDataFrame({var:wv[var]},geometry=gpd.points_from_xy(wv.x,wv.y),crs='EPSG:4326')
        print('gdf created')
        nodes=nodes.dropna()
        if clipper:
            clper = shp.gp.asGDF(clipper)
            nodes = nodes.to_crs(clper.crs)

            clpnodes = nodes.clip(
                clper.buffer(5280*clipbuff)
                )
        else:
            clpnodes = nodes
        if outvec:
            outvec.parent.mkdir(parents=True,exist_ok=True)
            clpnodes.to_file(outvec)
        cbe = make_geocube(clpnodes,output_crs=nodes.crs,interpolate_na_method='linear',resolution=(-res,res))
        print(f'{nc.parent.name} rasterized')
        if outTIF:
            outTIF.parent.mkdir(parents=True,exist_ok=True)
            cbe.rio.to_raster(outTIF)
            print(f'Bounced to {outTIF}')

        return cbe

class h5:
    def asHDF(hdf,mode='r'):
        '''if h5py.File, returns hdf\n
        else\n
        opens hdf in mode=mode and returns h5py.File'''
        if isinstance(hdf,h5py.File):
            return hdf
        else:
            return h5py.File(hdf,mode)
    def scan(openHDFfile,srchlen=None,srchTerm='',level=0):
        """prints structure of hdf5 file    \n
        openHDFfile object can be obtained with: h5py.File(hdfPath)\n
        if srchTerm, will print only items who have that str somewhere in their hierarchy\n
        if srchlen, will print only items whose # columns or # rows matches this int\n
        (This is helpful for finding which attributes corresponds to 2D flow area cells, faces, or facepoints)\n
        Do not specify level, used for searching recursively
        """
        #printhdfstructure
        for key in openHDFfile.keys():
            if isinstance(openHDFfile[key], h5py._hl.dataset.Dataset):
                if srchlen:
                    srch = ( openHDFfile[key].shape[0]==srchlen or(len(openHDFfile[key].shape)>1 and openHDFfile[key].shape[1]==srchlen) )
                else:
                    srch = True
                if srch:
                    if srchTerm in openHDFfile[key].name:
                        print(f"{'  '*level}  {openHDFfile[key].shape} DATASET: {openHDFfile[key].name}")
            elif isinstance(openHDFfile[key], h5py._hl.group.Group):
    #             print(f"{'  '*level} GROUP: {key, openHDFfile[key].name}")
                level += 1
                h5.scan(openHDFfile[key],srchlen=srchlen,srchTerm=srchTerm,level=level)
                level -= 1

            if openHDFfile[key].parent.name == "/":
                print("\n"*2)
    def copyGroup(fromHDF,toHDF,grup='Geometry'):
        '''TODO HDF's can be a subgroup rather than file path\n
        copy one to another, overwriting any existing '''

        phdf,ghdf = h5py.File(toHDF,'a') , h5py.File(fromHDF)

        with h5py.File(toHDF,  "a") as f:
            try:    
                del f[grup]
            except KeyError: # if it didn't exist
                pass
            
        assert ghdf[grup]
        # Group.copy() can certainly copy between files; just supply a File or
        # Group instance as the "dest" parameter:
        ghdf.copy(grup,phdf)

        assert phdf[grup]
        phdf.close()
        ghdf.close()
