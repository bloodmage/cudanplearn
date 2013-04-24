import dis

def listbytecode(co):
    code=co.co_code
    labels=dis.findlabels(code)
    n=len(code)
    i=0
    extended_arg=0
    free=None
    codes=[]
    cnt=0
    while i<n:
        c=code[i]
        op=ord(c)
        script={'pos':cnt,'line':i,'co':op,'code':dis.opname[op]}
        cnt+=1
        i=i+1
        codes.append(script)
        if op>dis.HAVE_ARGUMENT:
            oparg=ord(code[i])+ord(code[i+1])*256+extended_arg
            extended_arg=0
            i=i+2
            if op==dis.EXTENDED_ARG:
                extended_arg=oparg*65536L
            script['arg']=oparg
            if op in dis.hasconst:
                script['const']=co.co_consts[oparg]
            elif op in dis.hasname:
                script['name']=co.co_names[oparg]
            elif op in dis.hasjrel:
                script['target']=i+oparg
            elif op in dis.haslocal:
                script['local']=co.co_varnames[oparg]
            elif op in dis.hascompare:
                script['cmp']=dis.cmp_op[oparg]
            elif op in dis.hasfree:
                if free is None:
                    free=co.co_cellvars+co.co_frevars
                script['free']=free[oparg]
    print '\n'.join(map(repr,codes))
    return codes

class OPER:
    def __init__(self,name,*ops):
        self.name=name
        self.ops=ops
class NAME:
    def __init__(self,name,*ops):
        self.name=name
        self.ops=ops
class DETOUR:
    def __init__(self,val):
        self.val=val

EMPTYFLOW=lambda:[]
def INVALIDFLOW():raise Exception('Invalid flow')
def MAKETUPLE(*val): OPER('tuple',*val)
def MAKELIST(*val): OPER('list',*val)

Free_p=None
Name_p=None
Cmp_p=None

FLOWCONFIG={
    'STOP_CODE':        (0,EMPTYFLOW),
    'POP_TOP':          (1,lambda x:[]),
    'ROT_TWO':          (2,lambda x,y:[y,x]),
    'ROT_THREE':        (3,lambda x,y,z:[y,z,x]),
    'DUP_TOP':          (1,lambda x:[x,x]),
    'ROT_FOUR':         (4,lambda x,y,z,w:[y,z,w,x]),
    'NOP':              (0,EMPTYFLOW),
    'UNARY_POSITIVE':   (1,lambda x:[OPER('+',x)]),
    'UNARY_NEGATIVE':   (1,lambda x:[OPER('-',x)]),
    'UNARY_NOT':        (1,lambda x:[OPER('!',x)]),
    'UNARY_CONVERT':    (1,lambda x:[OPER('`',x)]),
    'UNARY_INVERT':     (1,lambda x:[OPER('~',x)]),
    'GET_ITER':         (1,lambda x:[OPER('iter',x)]),
    'BINARY_POWER':     (2,lambda x,y:[OPER('**',y,x)]),
    'BINARY_MULTIPLY':  (2,lambda x,y:[OPER('*',y,x)]),
    'BINARY_DIVIDE':    (2,lambda x,y:[OPER('/',y,x)]),
    'BINARY_FLOOR_DIVIDE':(2,lambda x,y:[OPER('//',y,x)]),
    'BINARY_TRUE_DIVIDE':(2,lambda x,y:[OPER('/',y,x)]),
    'BINARY_MODULO':    (2,lambda x,y:[OPER('%',y,x)]),
    'BINARY_ADD':       (2,lambda x,y:[OPER('+',y,x)]),
    'BINARY_SUBTRACT':  (2,lambda x,y:[OPER('-',y,x)]),
    'BINARY_SUBSCR':    (2,lambda x,y:[OPER('[]',y,x)]),
    'BINARY_LSHIFT':    (2,lambda x,y:[OPER('<<',y,x)]),
    'BINARY_RSHIFT':    (2,lambda x,y:[OPER('>>',y,x)]),
    'BINARY_AND':       (2,lambda x,y:[OPER('&',y,x)]),
    'BINARY_XOR':       (2,lambda x,y:[OPER('^',y,x)]),
    'BINARY_OR':        (2,lambda x,y:[OPER('|',y,x)]),
    'INPLACE_POWER':    (2,lambda x,y:[OPER('**',y,x)]),
    'INPLACE_MULTIPLY': (2,lambda x,y:[OPER('*',y,x)]),
    'INPLACE_DIVIDE':   (2,lambda x,y:[OPER('/',y,x)]),
    'INPLACE_FLOOR_DIVIDE':(2,lambda x,y:[OPER('//',y,x)]),
    'INPLACE_TRUE_DIVIDE':(2,lambda x,y:[OPER('/',y,x)]),
    'INPLACE_MODULO':   (2,lambda x,y:[OPER('%',y,x)]),
    'INPLACE_ADD':      (2,lambda x,y:[OPER('+',y,x)]),
    'INPLACE_SUBTRACT': (2,lambda x,y:[OPER('-',y,x)]),
    'INPLACE_LSHIFT':   (2,lambda x,y:[OPER('<<',y,x)]),
    'INPLACE_RSHIFT':   (2,lambda x,y:[OPER('>>',y,x)]),
    'INPLACE_AND':      (2,lambda x,y:[OPER('&',y,x)]),
    'INPLACE_XOR':      (2,lambda x,y:[OPER('^',y,x)]),
    'INPLACE_OR':       (2,lambda x,y:[OPER('|',y,x)]),
    'SLICE+0':          (1,lambda x:[OPER('[:]',x)]),
    'SLICE+1':          (2,lambda x,y:[OPER('[a:]',y,x)]),
    'SLICE+2':          (2,lambda x,y:[OPER('[:a]',y,x)]),
    'SLICE+3':          (3,lambda x,y,z:[OPER('[a:a]',z,y,x)]),
    'STORE_SLICE+0':    (2,lambda x,y:[]),
    'STORE_SLICE+1':    (3,lambda x,y,z:[]),
    'STORE_SLICE+2':    (3,lambda x,y,z:[]),
    'STORE_SLICE+3':    (4,lambda x,y,z,w:[]),
    'DELETE_SLICE+0':   (1,lambda x:[]),
    'DELETE_SLICE+1':   (2,lambda x,y:[]),
    'DELETE_SLICE+2':   (2,lambda x,y:[]),
    'DELETE_SLICE+3':   (3,lambda x,y,z:[]),
    'STORE_SUBSCR':     (3,lambda x,y,z:[]),
    'DELETE_SUBSCR':    (2,lambda x,y:[]),
    'PRINT_EXPR':       (1,lambda x:[]),
    'PRINT_ITEM':       (1,lambda x:[]),
    'PRINT_ITEM_DO':    (2,lambda x,y:[]),
    'PRINT_NEWLINE':    (0,EMPTYFLOW),
    'PRINT_NEWLINE_TO': (1,lambda x:[]),
    'BREAK_LOOP':       (0,EMPTYFLOW),      #CONTROL
    'CONTINUE_LOOP':    (0,EMPTYFLOW),      #CONTROL
    'LIST_APPEND':      (1,lambda x,y:[OPER('append',y,x)]),
    'LOAD_LOCALS':      (0,INVALIDFLOW),    #INVALID
    'RETURN_VALUE':     (1,lambda x:[]),    #CONTROL
    'YIELD_VALUE':      (0,INVALIDFLOW),    #INVALID
    'IMPORT_STAR':      (0,INVALIDFLOW),    #INVALID
    'EXEC_STMT':        (0,INVALIDFLOW),    #INVALID
    'POP_BLOCK':        (0,EMPTYFLOW),      #CONTROL
    'END_FINALLY':      (0,EMPTYFLOW),      #CONTROL
    'BUILD_CLASS':      (0,INVALIDFLOW),    #INVALID
    'SETUP_WITH':       (0,INVALIDFLOW),    #INVALID
    'WITH_CLEANUP':     (0,INVALIDFLOW),    #INVALID
    'STORE_NAME':       (1,lambda x:[]),
    'DELETE_NAME':      (0,EMPTYFLOW),
    'UNPACK_SEQUENCE':  (1,lambda x:[OPER('[]',x,i) for i in range(Target_p-1,-1,-1)]),
    'DUP_TOPX':         (1,lambda x:[x]*Free_p),
    'STORE_ATTR':       (0,INVALIDFLOW),    #INVALID
    'DELETE_ATTR':      (0,INVALIDFLOW),    #INVALID
    'STORE_GLOBAL':     (0,INVALIDFLOW),    #INVALID
    'DELETE_GLOBAL':    (0,INVALIDFLOW),    #INVALID
    'LOAD_CONST':       (0,lambda:[CONST(Const_p)]),
    'LOAD_NAME':        (0,lambda:[NAMED(Name_p)]),
    'BUILD_TUPLE':      (Free_p,MAKETUPLE),
    'BUILD_LIST':       (Free_p,MAKELIST),
    'BUILD_MAP':        (0,INVALIDFLOW),    #INVALID
    'LOAD_ATTR':        (1,lambda x:[OPER('attr',x,Name_p)]),
    'COMPARE_OP':       (2,lambda x,y:[OPER('b:%s'%Cmp_p,y,x)]),
    'IMPORT_NAME':      (0,EMPTYFLOW),
    'IMPORT_FROM':      (0,EMPTYFLOW),
    'JUMP_FORWARD':     (0,EMPTYFLOW),      #CONTROL
    'POP_JUMP_IF_TRUE': (1,lambda x:[],1,lambda x:[]),
    'POP_JUMP_IF_FALSE':(1,lambda x:[],1,lambda x:[]),
    'JUMP_IF_TRUE_OR_POP':(1,lambda x:[],0,EMPTYFLOW),
    'JUMP_IF_FALSE_OR_POP':(1,lambda x:[],0,EMPTYFLOW),
    'JUMP_ABSOLUTE':    (0,EMPTYFLOW),
    'FOR_ITER':         (1,lambda x:[OPER('it',x),x]),
    'LOAD_GLOBAL':      (0,lambda:[NAME('global',Name_p)]),
    'SETUP_LOOP':       (0,EMPTYFLOW),
    'SETUP_EXCEPT':     (0,INVALIDFLOW),
    'SETUP_FINALLY':    (0,INVALIDFLOW),
    'STORE_MAP':        (0,INVALIDFLOW),
    'LOAD_FAST':        (0,lambda:[NAME('name',Name_p)]),
    'STORE_FAST':       (1,lambda x:[]),
    'DELETE_FAST':      (0,EMPTYFLOW),
    'LOAD_CLOSURE':     (0,INVALIDFLOW),
    'LOAD_DEREF':       (0,INVALIDFLOW),
    'STORE_DEREF':      (0,INVALIDFLOW),
    'RAISE_VARARGS':    (0,INVALIDFLOW),
    'CALL_FUNCTION':    (Free_p,lambda *x:[OPER('call',*x)])
}
def flowdeduction(codelist):
    posmap={}
    for i in codelist:
        i['from']=[i['pos']-1]
        posmap[i['line']]=i

    for i in codelist:
        if i['code'] in 'CONTINUE_LOOP JUMP_FORWARD POP_JUMP_IF_TRUE POP_JUMP_IF_FALSE JUMP_IF_TRUE_OR_POP JUMP_IF_FALSE_OR_POP JUMP_ABSOLUTE FOR_ITER'.split(' '):
            i['jmp']=posmap[i['target'] if 'target' in i else i['arg']]['pos']
            posmap[i['target'] if 'target' in i else i['arg']]['from'].append(i['pos'])
        if i['code'] in 'JUMP_FORWARD CONTINUE_LOOP JUMP_ABSOLUTE'.split(' '):
            if i['arg']!=codelist[i['pos']+1]:
                codelist[i['pos']+1]['from'].remove(i['pos'])
    
    for i in codelist:
        i['from']=list(set(i['from']))


def typededuction(codelist,co,intypes):
    for i in codelist:
        i['flags']=set()
   
    def typeof(item):
        return item
    def mergestack(code,local,stack,stackblock):
        if 'stack' not in code:
            code['stack']=list(stack)
            code['stackblock']=list(stackblock)
            code['local']=local.copy()
            return True
        else:
            #Test mergeable
            updating=False
            if len(stack)!=len(code['stack']):
                raise Exception("Stack length mismatch")
            if len(stackblock)!=len(code['stackblock']):
                raise Exception("Block length mismatch")
            
            for i in range(len(stack)):
                if stack[i]!=code['stack'][i]:
                    updating=True


    def linededuct(linenum,linefrom,local,stack,stackblock):
        #Setup effects
        codelist[linenum]['flags'].add(linefrom)
        if codelist[linenum]['code']=='SETUP_LOOP':
            stackblock.append(len(stack))
        elif codelist[linenum]['code']=='POP_BLOCK':
            backblockpos=stackblock.pop()
            backblock=stack[backblockpos:]
            del stack[backblockpos:]
        #Run code & get type
        #run & deduce stack variables
        if codelist[linenum]['code'] in 'CONTINUE_LOOP JUMP_FORWARD JUMP_ABSOLUTE'.split(' '):
            #Single way
            if mergestack(codelist[linenum],local,stack,stackblock):
                linededuct(codelist[linenum]['jmp'],linenum,local,stack,stackblock)
        elif codelist[linenum]['code'] in 'POP_JUMP_IF_TRUE POP_JUMP_IF_FALSE'.split(' '):
            if typeof(stack[-1])!='b':
                raise Exception("POP_JUMP should consume a boolean on stack")
            v=stack.pop()
            if mergestack(codelist[linenum],local,stack,stackblock):
                linededuct(linenum+1,linenum,local,stack,stackblock)
                linededuct(codelist[linenum]['jmp'],linenum,local,stack,stackblock)
            stack.append(v)
        elif codelist[linenum]['code'] in 'JUMP_IF_TRUE_OR_POP JUMP_IF_FALSE_OR_POP'.split(' '):
            if typeof(stack[-1])!='b':
                raise Exception("POP_JUMP should consume a boolean on stack")
            if mergestack(codelist[linenum],local,stack,stackblock):
                linededuct(codelist[linenum]['jmp'],linenum,local,stack,stackblock)
                #Detour
                v=stack.pop()
                linededuct(linenum+1,linenum,local,stack,stackblock)
                stack.append(v)
        elif codelist[linenum]['code']=='FOR_ITER':
            if typeof(stack[-1])!='iter(i4)':
                raise Exception("FOR_ITER should peek a iter(i4)")
            if mergestack(codelist[linenum],local,stack,stackblock):
                stack.append('i4')
                linededuct(linenum+1,linenum,local,stack,stackblock)
                stack.pop()
                v=stack.pop()
                linededuct(codelist[linenum]['jmp'],linenum,local,stack,stackblock)
                stack.append(v)
        elif codelist[linenum]['code']=='RETURN_VALUE':
            mergestack(codelist[linenum],local,stack,stackblock)
            pass #Endroute
        elif codelist[linenum]['code']=='STORE_FAST':
            if codelist[linenum]['local'] in local:
                l=local[codelist[linenum]['local']]
                local[codelist[linenum]['local']]=stack[-1]
                exist=True
            else:
                exist=False
            if mergestack(codelist[linenum],local,stack,stackblock):
                linededuct(linenum+1,linenum,local,stack,stackblock)
            if exist:
                local[codelist[linenum]['local']]=l
            else:
                del local[codelist[linenum]['local']]
        else:
            #Normal codes
            cnt,func=FLOWCONFIG[codelist[linenum]['code']]
            if cnt!=0:
                sblk=stack[-cnt:]
                del stack[-cnt:]
            else:
                sblk=[]
            sblk.reverse()
            ans=func(*sblk)
            ans.reverse()
            stack.extend(ans)
            if mergestack(codelist[linenum],local,stack,stackblock):
                linededuct(linenum+1,linenum,local,stack,stackblock)
            if len(ans)!=0:
                del stack[-len(ans):]
                sblk.reverse()
            stack.extend(sblk)


        #Restore
        if codelist[linenum]['code']=='SETUP_LOOP':
            stackblock.pop()
        elif codelist[linenum]['code']=='POP_BLOCK':
            stackblock.append(backblockpos)
            stack.extend(backblock)
        


#Valid types: str b i1 i4 f4 f8 foreign iter(i4)
def typededuction(codelist,co,intypes):
    stack=[]
    #Type results
    name={}
    for i in range(co.co_argcount):
        name[co.co_varnames[i]]=inttypes

    for i in codelist:
        stack,func=FLOWCONFIG[i['code']]
        laststack=list(i['stack']) if i['from'][0]!=-1 else []
        #Run stack deduction
        if stack!=0:
            stackblock=laststack[-stack:]
            del laststack[-stack:]
        else:
            stackblock=[]
        
        stackblock.reverse()
        answer=func(*stackblock)
        answer.reverse()
        stackblock.extend(answer)
        
