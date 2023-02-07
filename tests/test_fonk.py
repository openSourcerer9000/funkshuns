#!/usr/bin/env python

"""Tests for `funkshuns` package."""

import pytest
from funkshuns.funkshuns import *

from jinja2 import FileSystemBytecodeCache
import pytest
# from contextlib import contextmanager

from pytest import approx
import pandas as pd, geopandas as gpd
import numpy as np
from pathlib import PurePath,Path
import os
import shutil
import h5py
#TODO fresh(x) @
#https://www.youtube.com/watch?v=2R1HELARjUk&feature=youtu.be
# many_triangles = [
#     (90, 60, 30, "right"),
#     (100, 40, 40, "obtuse"),
#     (60, 60, 60, "acute"),
#     (0, 0, 0, "invalid")
# ]

# @pytest.mark.s('a, b, c, expected', many_triangles)
# def test_func(a, b, c, expected):
#     assert triangle_type(a, b, c) == expected

#TODO organize into classes
#TODO split into separate files where long run tests can be run separate

mc = Path(os.path.dirname(os.path.abspath(__file__)))/'mockdata'
bkdir = mc/'fileBackup'
class mockstuff:
    def fresh(self,file):
        '''creates and returns the copy of the backup file\n
        the og does not have a _tst prefix
        '''
        freshest = file.parent/(file.stem+'_tst'+file.suffix)
        shutil.copy(file,freshest)
        return freshest
    def __init__(self):
            self.seq = ['Geom File=g03\n', 'Geom File=g04\n', 'Geom File=g05\n', 'Geom File=g02\n', 'Geom File=g09\n', 'Geom File=g08\n', 'Geom File=g11\n', 'Unsteady File=u01\n', 'Unsteady File=u02\n', 'Unsteady File=u03\n', 'Unsteady File=u04\n', 'Plan File=p02\n', 'Plan File=p08\n', 'Plan File=p09\n', 'Plan File=p03\n', 'Plan File=p07\n', 'Plan File=p04\n', 'Plan File=p10\n', 'Plan File=p11\n', 'Plan File=p12\n', 'Plan File=p13\n', 'Plan File=p14\n', 'Plan File=p15\n', 'Plan File=p16\n', 'Plan File=p17\n', 'Plan File=p18\n', 'Plan File=p19\n', 'Background Map Layer=culs\n']
            self.lines = ['Hello there','Nice to meet you','My name is','Goodbye']
            self.geo = mc/'geo.g01'
            self.msg = mc/'msg.txt'
            self.prj = mc/'proj.prj'
            self.p01,self.p02 = [ mc/f'proj.{p}' for p in ('p01','p02') ]
x=mockstuff()

# def test_2txts_are_equal():
#     '''not a unit test, just using the function with pytest comparison'''
#     pth = Path(r'C:\Users\sean.micek\Desktop\Galv\DickinsonBayouModeling09.08.2020\HEC-RAS')
#     assert txt.equal([pth/'Dickinson_Bayou.u05',pth/'Dickinson_Bayou - Copy.u05'])
def test_x():
    print(x.msg)
    assert x.lines == ['Hello there','Nice to meet you','My name is','Goodbye']
def test_xfresh():
    msg=x.fresh(x.msg)
    assert msg.exists()

def test_regresh():
    pass
def test_rem():
    pass #TODO
def test_rem_doesntexist():
    pass #TODO
def test_rems():
    msg=x.fresh(x.msg)
    msg2=x.fresh(msg)
    assert (msg).exists()
    assert (msg2).exists()
    rems([msg,msg2])
    assert not (msg2).exists()
    assert not (msg).exists()
def test_rems_backup():
    pass #TODO
def test_fileBackup_and_restore():
    filee = bkdir/'backDatThangUp.txt'
    filee.unlink(missing_ok=True)
    bkfilee = filee.parent/'Backup'/filee.name
    bkfilee.unlink(missing_ok=True)

    filee.write_text('booom')
    fileBackup(filee)

    assert bkfilee.read_text()=='booom'

    filee.unlink(missing_ok=True)
    filee.write_text('working that back')
    
    fileRestore(filee)

    assert filee.read_text()=='booom'
    assert bkfilee.read_text()=='working that back'
def test_unzipAll():
    #setup
    zip1,zip2 = mc/'zip1',mc/'zip2'
    if not zip1.exists():
        zip1.mkdir()
    if not zip2.exists():
        zip2.mkdir()
    zip1fles = [x.msg,x.prj]
    zip2fles =  [x.p01]
    [shutil.copy(fle,zip1/fle.name) for fle in zip1fles]
    [shutil.copy(fle,zip2/fle.name) for fle in zip2fles]
    zipd1,zipd2 = zip1.parent/f'{zip1.name}new',zip2.parent/f'{zip2.name}new'
    shutil.make_archive(zipd1, 'zip', zip1)
    shutil.make_archive(zipd2, 'zip', zip2)

    unzipAll(mc) #to zip1,zip2 pths
    unzipAll(mc,mc/'zipAs1pth')
    fles1 = list((mc/'zipAs1pth').glob('*'))
    fles2 = list(zipd1.glob('*'))+list(zipd2.glob('*'))
    fles1,fles2 = ( sorted([f.name for f in fles1]),sorted([f.name for f in fles2]) )
    assert fles1==fles2
    assert listisin(fles1,[f.name for f in zip1fles+zip2fles])
    #cleanup - why for loop returns err????
    shutil.rmtree(zip1)
    shutil.rmtree(zip2)
    shutil.rmtree(mc/'zipAs1pth')
    shutil.rmtree(zipd1)
    shutil.rmtree(zipd2)
    (mc/f'{zipd1.name}.zip').unlink()
    (mc/f'{zipd2.name}.zip').unlink()
def test_flattenlist():
    i=[ [1,2],[1,2,3,4] ]
    o=[1, 2, 1, 2, 3,4]
    assert flattenList(i) == o
def test_listJoin():
    i = [ (['a','b','c','d','e'],'-') , [x.lines[:]] ]
    o = [ ['a','-','b','-','c','-','d','-','e'] , ['Hello there','\n','Nice to meet you','\n','My name is','\n','Goodbye'] ]
    msg = [ '','multiple character strs' ]
    for j in range(len(i)):
        assert listJoin(*i[j]) == o[j], msg[j]
def test_dropSuff():
    i = (['a St','b Ct','c d place'],['a','b st', 'c d st'])
    o = [['a','b','c d']]*2
    for ii,oo in zip(i,o):
        assert dropSuff(ii)==oo 
def test_dropSuff_3():
    i = (['a St pl ct','b Ct st','c d place st ave'],['a 1 sdf 4','b', 'c d st pl 34_+/*-/*-'])
    o = [['a','b','c d']]*2
    for ii,oo in zip(i,o):
        assert dropSuff(ii,3)==oo 
def test_dropSuff_delim():
    i = (['ax Q q','bxd','c dx3'] , ['a','bx4','c dx st pl 34_+/*-/*-'])
    o = [['a','b','c d']]*2
    for ii,oo in zip(i,o):
        assert dropSuff(ii,delim='x')==oo 
def test_dropSuff_strlistproblem():
    i = ['apple banana','blueberry','cranberry koolaid']
    o = ['apple','blueberry','cranberry']
    assert dropSuff(i)==o
def test_findNth():
    st = r'Connection Culv=shpe,rise,span,football'
    assert st[findNth(',',st,3):]==',football'

def test_strChunk():
    i = [ (r'123456789',3) , (r'vincent',5) ]
    o = [ ['123','456','789'], ['vince','nt'] ]
    for j in range(len(i)):
        assert strChunk(*i[j]) == o[j]
def test_listChunk_num():
    i = list(range(13))
    i = [j*111 for j in i]
    o = [ [0,111,222,333,444],[555,666,777,888,999],[10*111,11*111,12*111] ]
    assert listChunk(i,5) == o
def test_listChunk_str():
    i = x.lines[:]
    o = [ ['Hello there','Nice to meet you'],['My name is','Goodbye'] ]
    # print(listChunk(i,2))
    assert listChunk(i,2) == o
def test_listisin():
    i = [[1,2,3],[5,3,4,2,1]]
    assert listisin(*i)
def test_assertListisin():
    with pytest.raises(AssertionError):
        assertListisin(['bill','joe'],['bill','joey','bob'])
def test_assertListisin_warning():
    assertListisin(['bill','joe'],['bill','joey','bob'],lethal=False)
def test_listInsertEveryNth():
    i = ([1,2,3]*5,['aa','bb']*3)
    iN_what = ((4,0),(3,'cc'))
    o = ([1,2,3,0]*5,['aa','bb','cc']*3)
    for ii,iNW,oo in zip(i,iN_what,o):
        assert listInsertEveryNth(ii,*iNW)==oo
def test_listrip_inplace():
    lst = ['','','','','',1,1,2,'','']
    listrip(lst,'')
    assert lst==[1,1,2]
def test_listrip_return():
    lst = ['','','','','',1,'1',2,'','']
    assert listrip(lst,'')==[1,'1',2]
def test_listEqual():
    assert listEqual([4]*5)
    assert listEqual(['abcd']*3)
    assert not listEqual([1,1,1,1.1])
def test_isiter():
    # non-string iterables
    assert isiter(("f", "f"))    # tuple
    assert isiter(["f", "f"])    # list
    assert isiter(iter("ff"))    # iterator
    assert isiter(range(44))     # generator

    # strings, pd stuff
    assert not isiter(b"yte string")         # bytes (Python 2 calls this a string)
    assert not isiter(u"ff")     # string
    assert not isiter(44)        # integer
    assert not isiter(isiter)  # function
    assert not isiter(pd.DataFrame)
    assert not isiter(pd.Series)
    assert not isiter(gpd.GeoDataFrame)
    assert not isiter(gpd.GeoSeries)
def test_asList():
    arg = lambda x: x**2
    newvar = asList(arg)
    assert [i for i in newvar]
def test_asList_list():
    arg = ['one','two','three']
    newvar = asList(arg)
    assert [i for i in newvar]
def test_asList_copies():
    arg = [1]
    newvar = asList(arg)
    newvar += [2]
    assert newvar!=arg
def test_dictSwap():
    assert dictSwap({'a':1,'b':2,'ccc':333}) == {1:'a',2:'b',333:'ccc'}
def test_ABtree():    
    A = np.array([[0,1],[1,2],[3,4],[5,6]])
    B = np.array([[1,2],[3,4],[0,1],[667,800]])
    assert (xnp.ABtree(A,B,2) == np.array([ 1,  2,  0, -1])).all()
def test_Atree():
    A = np.array([[0,1],[0,1],[1,2],[1,2],[3,4]])
    assert (xnp.Atree(A,2) == np.array([1,0,3,2,-1])).all()
@pytest.mark.parametrize('A,B,o',[
    (np.array([[0,1],[1,2],[3,4],[5,6]]),np.array([[1,2],[3,4],[0,1],[667,800]]),
        np.array([ 1,  2,  0, -1]) ),
    (np.array([[0,1],[0,1],[1,2],[1,2],[3,4]]),np.array([[0,1],[0,1],[1,2],[1,2],[3,4]]),
        np.array([1,0,3,2,-1]))
    ])    
def test_KDTreee(A,B,o):
    assert (xnp.KDTreee(A,B,2) == o).all()
    
@pytest.mark.parametrize('i,o',[
    ('hello world','hello_world'),
    ('9hello^world','_9hello_world'),
    ('hello.world','hello_world')
    ])    
def test_ondah(i,o):
    assert ondah(i)==o

def test_xnp_replace():
    ten = np.arange(10)
    d = dict(zip(ten,ten+1))
    assert (xnp.replace(ten,d) == ten+1).all()

    five = np.arange(5)
    assert (xnp.replace(five,d) == five+1).all()

    mo = np.arange(10,30)
    i = np.concatenate([five,mo])
    o = np.concatenate([five+1,mo])
    assert (xnp.replace(i,d) == o).all()

def test_unByte():
    i = pd.DataFrame([[b'aaa',b'b'],[b'c',b'ddd']])
    o = pd.DataFrame([['aaa','b'],['c','ddd']])
    assert xpd.unByte(i).equals(o)
def test_regx():
    assert regx(regxASCII,r'123456789move01 23 456') == 'move'
def test_extractNum():
    i ='asdfd67fds5850'
    o = 67
    assert extractNum(i) ==o
def test_extractNum_decimal():
    io ={'asdfd67.05fds5850':67.05,'-0.35fdx':-0.35,'e5.5.5x':5.5,'868868686868xx':868868686868,'x9d8e5':9}
    assert [extractNum(i) for i in list(io.keys()) ] == list(io.values()) #TODO func this
def test_reNestedSub():
    red = '\nBEGIN DESCRIPTION:\nsome line\nEND DESCRIPTION:\nline\nline\nBEGIN DESCRIPTION:\nsome other line\nEND DESCRIPTION:\nline\n'
    reggie = re.compile('(\nBEGIN DESCRIPTION:.*?END DESCRIPTION:)',re.DOTALL)
    o = '\nbb BEGIN DESCRIPTION:\nbb some line\nbb END DESCRIPTION:\nline\nline\nbb BEGIN DESCRIPTION:\nbb some other line\nbb END DESCRIPTION:\nline\n'
    assert  reNestedSub(red,reggie,'\n','\nbb ',re.DOTALL)==o

class Testxpd():
    @pytest.mark.parametrize('df', [
        ( pd.DataFrame.from_dict({'idx': {0: 'a', 1: 'a', 2: 'b', 3: 'b', 4: 'b', 5: 'c'},
            'w8': {0: 0, 1: 1, 2: 7, 3: 7, 4: 7, 5: 3},
            'val1': {0: 10, 1: 5, 2: 7, 3: 5, 4: 3, 5: 5},
            'val2': {0: 20, 1: 10, 2: 15, 3: 10, 4: 5, 5: 10}}).set_index('idx') ),
        ( pd.DataFrame.from_dict({'idx': {0: 'a', 1: 'a', 2: 'b', 3: 'b', 4: 'b', 5: 'c'},
        'w8': {0: 5, 1: 5, 2: 7, 3: 7, 4: 7, 5: 3},
        'val1': {0: np.nan, 1: 5.0, 2: 7.0, 3: 5.0, 4: 3.0, 5: 5.0},
        'val2': {0: 8, 1: 12, 2: 15, 3: 10, 4: 5, 5: 10}}).set_index('idx') ),
        ( pd.DataFrame.from_dict({'idx': {0: 'a', 1: 'a', 2: 'b', 3: 'b', 4: 'b', 5: 'c'},
            'idx2':dict(zip(range(6),range(6))),
        'w8': {0: 5, 1: 5, 2: 7, 3: 7, 4: 7, 5: 3},
        'val1': {0: np.nan, 1: 5.0, 2: 7.0, 3: 5.0, 4: 3.0, 5: 5.0},
        'val2': {0: 8, 1: 12, 2: 15, 3: 10, 4: 5, 5: 10}}).set_index(['idx','idx2']) )
        ]
    )
    def test_weightedAvg(self,df):
        avg = xpd.weightedAvg(df,'w8')
        assert avg.val1.to_list() == [5]*3
        assert avg.val2.to_list() == [10]*3
        assert avg.reset_index().to_dict() == {'idx': {0: 'a', 1: 'b', 2: 'c'}, 'val1': {0: 5.0, 1: 5.0, 2: 5.0}, 'val2': {0: 10.0, 1: 10.0, 2: 10.0}}
    
    def test_weightedAvg_allNaNs(self):
        df = pd.DataFrame.from_dict(
            {'idx': {0: 'a', 1: 'a', 2: 'b', 3: 'b', 4: 'b', 5: 'c'},
            'w8': {0: 5, 1: 5, 2: 7, 3: 7, 4: 7, 5: 3},
            'val1': {0: np.nan, 1: np.nan, 2: 7.0, 3: 5.0, 4: 3.0, 5: np.nan},
            'val2': {0: 8, 1: 12, 2: 15, 3: 10, 4: 5, 5: 10}}).set_index('idx')
        avg = xpd.weightedAvg(df,'w8')
        assert avg.val1.isna().to_list() == [True,False,True]
        assert avg.val2.to_list() == [10]*3

    HDF_dfs = [
        ( 
            pd.DataFrame.from_dict({
            'w8': {0: 0, 1: 1, 2: 7, 3: 7, 4: 7, 5: 3},
            'val1': {0: 10, 1: 5, 2: 7, 3: 5, 4: 3, 5: 5},
            'val2': {0: 20, 1: 10, 2: 15, 3: 10, 4: 5, 5: 10}}) 
        ),
        (
            pd.DataFrame(np.random.rand(5,4))
        ),
        ]

    @pytest.mark.parametrize('df', HDF_dfs)
    def test_tohdf_readhdf(self,df):
        hdffile = mc/'ignore'/'test.hdf'
        hdf = h5py.File(hdffile,'w')

        xpd.tohdf(df,hdf,'ds')

        Hdf = xpd.readhdf(hdf['ds'])
        hdf.close()
        assert Hdf.equals(df)

    @pytest.mark.parametrize('df', HDF_dfs)
    def test_tohdf_overwrite(self,df):
        hdffile = mc/'ignore'/'test.hdf'

        hdf = h5py.File(hdffile,'w')
        OGdf = pd.DataFrame(np.random.rand(1,2))
        xpd.tohdf(OGdf,hdf,'ds')
        hdf.close()

        hdf = h5py.File(hdffile,'a')
        xpd.tohdf(df,hdf,'ds')

        Hdf = xpd.readhdf(hdf['ds'])
        hdf.close()
        assert Hdf.equals(df)



def test_replaceMulti_happens_in_one_pass():
    i = ( {'Johnson':'Cobham','Joe':'Paul','Paul':'Seymour'},'Billy Johnson, Joe Grisham, Paul Witzer, Joe Johnson' )
    o = 'Billy Cobham, Paul Grisham, Seymour Witzer, Paul Cobham'
    assert replaceMulti(*i) == o
