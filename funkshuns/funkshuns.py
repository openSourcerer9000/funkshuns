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
    percent_done = math.ceil(percent_done*10)/10

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
def cmder(cmd,*args):
    '''pass to cmd line and print output from py\n
    cmd: str 
    returns time elapsed, TODO allow returns from cmd\n
    TODO kwargs passed to cmd
    https://stackoverflow.com/a/6414278/13030053'''
    thenn = datetime.now()
    p = subprocess.Popen([cmd,*args], stdout=subprocess.PIPE, bufsize=1)
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
    if filee.suffix =='.shp':
        filees = filee.parent.glob(f'{filee.stem}.*')
        [ shutil.copy(filee,pthBackup/filee.name) for filee in filees ]
    else:
        shutil.copy(filee,pthBackup/filee.name)
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

try:
    import requests
except Exception as e:
    if printImportWarnings:
        print('WARNING: ',e)


def DL(url,toPth):
    '''DL's file from url and places in toPth Path\n
    returns request status code'''
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
    return r.status_code

from multiprocessing.pool import ThreadPool

@timeit
def DLmulti(urls,toPth,threds=12):
    results = ThreadPool(threds).starmap(DL, [(url,toPth) for url in urls])
    return list(zip(urls,results))

import zipfile
@timeit
def unzipAll(pthOfZips,to1pth=None):
    '''unzips all .zip's in pthOfZips to individual folders\n
    if to1pth: unzip all to Path to1pth instead'''
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

from functools import reduce
from operator import iconcat
def flattenList(listOfLists):
    '''or list of tups'''
    return reduce(iconcat,listOfLists,[])
def globdown(pth,extensh,lvls):
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
import collections
def isiter(obj):
    '''True if iterable, False if string type, b'string', (geo)pandas obj, or not iterable'''
    return (
        isinstance(obj, collections.Iterable) 
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
from inspect import getargspec
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

class xnp():
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
class xpd():
    '''Handy tools for pandas'''
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
    def readhdf(hdf):
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
        return df
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

class meta():
    '''metaprogramming utils'''
    def functionize(file=Path(r'C:\Users\seanm\Desktop\temp')/"Untitled-1.py" ,iniCells=3,argCell=2,src='vscode'):
        '''Functionizes a .ipynb Exported as executable script (.py)\n
        iniCells: how many code cells before the function begins\n
        argCell: which cell to pull args from'''
        key = '# %%' if src=='vscode' else '# In['
        tabb = ' '*4
        def deJpy(lines):
            '''cleans the jupyter out of Exported as executable scripts (.py)'''
            p = [line for line in lines if not line.startswith(key)]
            wspace = []
            for i in range(len(p)):
                if not p[i-1:i+1]==['','']:
                    wspace += [ p[i] ]
            return wspace
        # p = file.read_text().split('\n')
        lines = read(file)
        lines[6:11]
        
        funkstart = findKey(lines,[key]*(iniCells+1))+1
        ini,funk = lines[:funkstart],lines[funkstart:]
        ini
        
        #del extra whitespace
        funk = funk[findKey(funk,r'\w+',how='regex'):]
        funk[:5]
        
        ini,funk = [deJpy(lnes) for lnes in (ini,funk)]
        ini
        
        #TODO convention for this? Just big bold docstrings?
        found = findKey(lines,'#  # ')
        if found:
            docstring = "'''" + lines[found].replace('#  # ','') + "'''"
        else:
            docstring = "''''''"
        print(docstring)
        
        argstart = findKey(lines,[key]*(argCell-1))+1
        lines[argstart:]
        
        argz = [i for i in ini if i.find('=')>-1]
        argz
        
        argz = [a.split('=')[0] for a in argz]
        argz
        
        splitit = lambda argz: [[a for a in arg.split(' ') if len(a)>0][0] for arg in argz]
        assert splitit(['arg1 ', 'arg2','   arg3']) == ['arg1', 'arg2','arg3']
        argz = splitit(argz)
        argz
        
        funk = [f'{tabb}{line}' for line in funk]
        funk[-5:]
        
        funknmer = lambda filestem: filestem.split('_')[-1].split(' ')[0]
        assert funknmer('w_PC_funk (6)')=='funk'
        assert funknmer('w_PC_funk')=='funk'
        funknm = funknmer(file.stem)
        funknm
        
        deff = f"def {funknm}({','.join(argz)}):"
        deff
        
        res = ini+[deff]+[tabb+docstring]+funk+[f'{tabb}return ']
        res
        
        file.write_text('\n'.join(res))
        print(f'{file} has been functionized')