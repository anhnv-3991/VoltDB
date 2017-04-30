import os, sys, threading, shutil
import subprocess # add part
from subprocess import Popen, PIPE, STDOUT

class BuildContext:
    def __init__(self, args):
        self.CPPFLAGS = ""
        self.EXTRAFLAGS = ""
        self.LDFLAGS = ""
        self.JNIEXT = ""
        self.JNILIBFLAGS = ""
        self.JNIBINFLAGS = ""
        self.SOFLAGS = ""
        self.SOEXT = ""
        self.IGNORE_SYS_PREFIXES = ()
        self.INPUT_PREFIX = ""
        self.THIRD_PARTY_INPUT_PREFIX = ""
        self.OUTPUT_PREFIX = ""
        self.TEST_PREFIX = ""
        self.INPUT = {}
        self.THIRD_PARTY_INPUT = {}
        self.TESTS = {}
        self.PLATFORM = os.uname()[0]
        self.LEVEL = "DEBUG"
        self.TARGET = "BUILD"
        self.NM = "/usr/bin/nm"
        self.NMFLAGS = "-n"    # specialized by platform in build.py
        self.COVERAGE = False
        self.PROFILE = False
        self.CC = "gcc"
        self.CXX = "g++"
        for arg in [x.strip().upper() for x in args]:
            if arg in ["DEBUG", "RELEASE", "MEMCHECK", "MEMCHECK_NOFREELIST"]:
                self.LEVEL = arg
            if arg in ["BUILD", "CLEAN", "TEST", "VOLTRUN", "VOLTDBIPC"]:
                self.TARGET = arg
            if arg in ["COVERAGE"]:
                self.COVERAGE = True
            if arg in ["PROFILE"]:
                self.PROFILE = True
        # Exec build.local if available, a python script provided with this
        # BuildContext object as the update-able symbol BUILD.  These example
        # build.local lines force the use of clang instead of gcc/g++.
        #   BUILD.CC = "clang"
        #   BUILD.CXX = "clang"
        buildLocal = os.path.join(os.path.dirname(__file__), "build.local")
        if os.path.exists(buildLocal):
            execfile(buildLocal, dict(BUILD = self))

def readFile(filename):
    "read a file into a string"
    FH=open(filename, 'r')
    fileString = FH.read()
    FH.close()
    return fileString

# get the volt version by reading buildstring.txt
version = "0.0.0"
try:
    version = readFile("version.txt")
    version = version.strip()
except:
    print "ERROR: Unable to read version number from version.txt."
    sys.exit(-1)

def getAllDependencies(inputs, threads, retval={}):
    if threads == 1:
        for filename, cppflags, sysprefixes in inputs:
            retval[filename] = getDependencies(filename, cppflags, sysprefixes)
    else:
        r2 = {}
        t1 = threading.Thread(None, getAllDependencies, "checkdeps1", (inputs[:len(inputs)/2], 1, retval))
        t2 = threading.Thread(None, getAllDependencies, "checkdeps2", (inputs[len(inputs)/2:], 1, r2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        retval.update(r2)
    return retval

def startsWithFromTuple(strValue, prefixes):
    for prefix in prefixes:
        if strValue.startswith(prefix):
            return True
    return False

def getDependencies(filename, cppflags, sysprefixes):
    command = "g++ %s -MM %s" % (cppflags, filename)

    pipe = Popen(args=command, shell=True, bufsize=-1, stdout=PIPE, stderr=PIPE)
    out = pipe.stdout.readlines()
    out_err = pipe.stderr.readlines()
    retcode = pipe.wait()
    if retcode != 0:
        print "Error Determining Dependencies for: %s" % (filename)
        print ''.join(out_err)
        sys.exit(-1)

    if len(out) > 0:
        out = out[1:]
    out = " ".join(out)
    out = out.split()
    out = [x.strip() for x in out]
    out = [x for x in out if x != "\\"]
    out = [x for x in out if not startsWithFromTuple(x, tuple(sysprefixes))]
    return out

def outputNamesForSource(filename):
    relativepath = "/".join(filename.split("/")[2:])
    jni_objname = "objects/" + relativepath[:-2] + "o"
    static_objname = "static_objects/" + relativepath[:-2] + "o"
    return jni_objname, static_objname

def namesForTestCode(filename):
    relativepath = "/".join(filename.split("/")[2:])
    binname = "cpptests/" + relativepath
    sourcename = filename + ".cpp"
    objectname = "static_objects/" + filename.split("/")[-1] + ".o"
    return binname, objectname, sourcename

def buildMakefile(CTX):
    global version

    CPPFLAGS = " ".join(CTX.CPPFLAGS.split())
    MAKECPPFLAGS = CPPFLAGS
    for dir in CTX.SYSTEM_DIRS:
        MAKECPPFLAGS += " -isystem ../../%s" % (dir)
    for dir in CTX.INCLUDE_DIRS:
        MAKECPPFLAGS += " -I../../%s" % (dir)
    LOCALCPPFLAGS = CPPFLAGS
    for dir in CTX.SYSTEM_DIRS:
        LOCALCPPFLAGS += " -isystem %s" % (dir)
    for dir in CTX.INCLUDE_DIRS:
        LOCALCPPFLAGS += " -I%s" % (dir)
    JNILIBFLAGS = " ".join(CTX.JNILIBFLAGS.split())
    JNIBINFLAGS = " ".join(CTX.JNIBINFLAGS.split())
    INPUT_PREFIX = CTX.INPUT_PREFIX.rstrip("/")
    THIRD_PARTY_INPUT_PREFIX = CTX.THIRD_PARTY_INPUT_PREFIX.rstrip("/")
    OUTPUT_PREFIX = CTX.OUTPUT_PREFIX.rstrip("/")
    TEST_PREFIX = CTX.TEST_PREFIX.rstrip("/")
    IGNORE_SYS_PREFIXES = CTX.IGNORE_SYS_PREFIXES
    JNIEXT = CTX.JNIEXT.strip()
    NM = CTX.NM
    NMFLAGS = CTX.NMFLAGS

#add part

#get compute capability that need .cu file compile to .cubin file
    subprocess.call(['make', '--directory=third_party/gpu/'])
    p = subprocess.Popen ("third_party/gpu/check_cc", shell=True, stdin=subprocess.PIPE,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      close_fds=True)
    ComCap = p.stdout.readline()
    if "A GPU which is capable of using CUDA is not detected..." in ComCap:
        return False

    GPUFLAGS = "-lcuda -lstdc++ -lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include -I/usr/local/cuda/include -I../../src/ee/executors/ -I../../src/ee/ -I/usr/local/cuda/samples/6_Advanced/ -I/usr/local/cuda/samples/common/inc"
    INCLUDE = "-isystem ../../third_party/cpp/ -lm"
    GPUARCHFLAGS = "-arch sm_%s "%ComCap
    LOCALCPPFLAGS += " %s %s" % (GPUFLAGS,INCLUDE)
#add part end

    # create directories for output if they don't exist
    os.system("mkdir -p %s" % (OUTPUT_PREFIX))
    os.system("mkdir -p %s" % (OUTPUT_PREFIX + "/nativelibs"))
    os.system("mkdir -p %s" % (OUTPUT_PREFIX + "/objects"))
    os.system("mkdir -p %s" % (OUTPUT_PREFIX + "/static_objects"))
    os.system("mkdir -p %s" % (OUTPUT_PREFIX + "/cpptests"))
    os.system("mkdir -p %s" % (OUTPUT_PREFIX + "/prod"))
#Added for GPU
    os.system("mkdir -p %s" % (OUTPUT_PREFIX + "/objects/GPUetc/indexes"))
#End

    input_paths = []
    for dir in CTX.INPUT.keys():
        input = CTX.INPUT[dir].split()
        input_paths += [INPUT_PREFIX + "/" + dir + "/" + x for x in input]

    third_party_input_paths = []
    for dir in CTX.THIRD_PARTY_INPUT.keys():
        input = CTX.THIRD_PARTY_INPUT[dir].split()
        third_party_input_paths += [THIRD_PARTY_INPUT_PREFIX + "/" + dir + "/" + x for x in input]

    tests = []
    for dir in CTX.TESTS.keys():
        input = CTX.TESTS[dir].split()
        tests += [TEST_PREFIX + "/" + dir + "/" + x for x in input]

    makefile = file(OUTPUT_PREFIX + "/makefile", 'w')
    makefile.write("CC = %s\n" % CTX.CC)
    makefile.write("CXX = %s\n" % CTX.CXX)
    makefile.write("CPPFLAGS += %s\n" % (MAKECPPFLAGS))
    makefile.write("GPUFLAGS += %s\n" % (GPUFLAGS))     #add part
    makefile.write("INCLUDE += %s\n" % (INCLUDE))       #add part
    makefile.write("GPUARCHFLAGS += %s\n" % (GPUARCHFLAGS))       #add part
    makefile.write("LDFLAGS += %s\n" % (CTX.LDFLAGS))
    makefile.write("JNILIBFLAGS += %s\n" % (JNILIBFLAGS))
    makefile.write("JNIBINFLAGS += %s\n" % (JNIBINFLAGS))
    makefile.write("JNIEXT = %s\n" % (JNIEXT))
    makefile.write("SRC = ../../%s\n" % (INPUT_PREFIX))
    makefile.write("THIRD_PARTY_SRC = ../../%s\n" % (THIRD_PARTY_INPUT_PREFIX))
    makefile.write("NM = %s\n" % (NM))
    makefile.write("NMFLAGS = %s\n" % (NMFLAGS))

    makefile.write("\n")




    if CTX.TARGET == "CLEAN":
        makefile.write(".PHONY: clean\n")
        makefile.write("clean: \n")
        makefile.write("\trm -rf *\n")
        makefile.close()
        return

    makefile.write(".PHONY: main\n")
    if CTX.TARGET == "VOLTRUN":
        makefile.write("main: prod/voltrun\n")
    elif CTX.TARGET == "TEST":
        makefile.write("main: ")
    else:
        makefile.write("main: nativelibs/libvoltdb-%s.$(JNIEXT)\n" % version)
    makefile.write("\n")

    jni_objects = []
    static_objects = []
    for filename in input_paths:
        jni, static = outputNamesForSource(filename)
        jni_objects.append(jni)
        static_objects.append(static)
    for filename in third_party_input_paths:
        jni, static = outputNamesForSource(filename)
        jni_objects.append(jni)
        static_objects.append(static)

    makefile.write("# create symbols by running nm against libvoltdb-%s\n" % version)
    makefile.write("nativelibs/libvoltdb-%s.sym: nativelibs/libvoltdb-%s.$(JNIEXT)\n" % (version, version))
    makefile.write("\t$(NM) $(NMFLAGS) nativelibs/libvoltdb-%s.$(JNIEXT) > $@\n" % version)
    makefile.write("\n")

    makefile.write("# main jnilib target\n")
#    makefile.write("nativelibs/libvoltdb-%s.$(JNIEXT): " % version + " ".join(jni_objects) + "\n")  # add part
    makefile.write("nativelibs/libvoltdb-%s.$(JNIEXT): " % version + " ".join(jni_objects) + " objects/executors/scan.co objects/executors/GPUNIJ.co objects/executors/GPUSHJ.co objects/executors/GPUIJ.co objects/executors/GPUHJ.co objects/executors/GPUINSERT.co objects/executors/index_join_gpu.co objects/executors/join_gpu.co objects/executors/ghash.co objects/executors/gcommon.co objects/GPUetc/indexes/TreeIndex.co objects/GPUetc/storage/gtable.co" + "\n")  # add part
    makefile.write("\t$(LINK.cpp) $(JNILIBFLAGS) $(GPUFLAGS) -o $@ $^\n") #add part
    makefile.write("\n")

    makefile.write("# voltdb instance that loads the jvm from C++\n")
    makefile.write("prod/voltrun: $(SRC)/voltrun.cpp " + " ".join(static_objects) + "\n")
    makefile.write("\t$(LINK.cpp) $(JNIBINFLAGS) -o $@ $^\n")
    makefile.write("\n")

    makefile.write("# voltdb execution engine that accepts work on a tcp socket (vs. jni)\n")
    makefile.write("prod/voltdbipc: $(SRC)/voltdbipc.cpp " + " objects/volt.a\n")
    makefile.write("\t$(LINK.cpp) -o $@ $^ %s\n" % (CTX.LASTLDFLAGS))
    makefile.write("\n")


    makefile.write(".PHONY: test\n")
    makefile.write("test: ")
    for test in tests:
        binname, objectname, sourcename = namesForTestCode(test)
        makefile.write(binname + " ")
    if CTX.LEVEL == "MEMCHECK":
        makefile.write("prod/voltdbipc")
    if CTX.LEVEL == "MEMCHECK_NOFREELIST":
        makefile.write("prod/voltdbipc")
    makefile.write("\n\n")
    makefile.write("objects/volt.a: " + " ".join(jni_objects) + " objects/harness.o\n")
    makefile.write("\t$(AR) $(ARFLAGS) $@ $?\n")
    makefile.write("objects/harness.o: ../../" + TEST_PREFIX + "/harness.cpp\n")
    makefile.write("\t$(CCACHE) $(COMPILE.cpp) -o $@ $^\n")
    makefile.write("\t$(CCACHE) $(COMPILE.cpp) -o $@ $^\n")
    makefile.write("\n")


#add part
    GPUPATH = "../../src/ee/executors/"
    GPUINC = " ../../src/ee/executors/GPUNIJ.h ../../src/ee/executors/GPUSHJ.h ../../src/ee/executors/GPUIJ.h ../../src/ee/executors/GPUINSERT.h ../../src/ee/executors/GPUHJ.h ../../src/ee/GPUetc/common/GPUTUPLE.h ../../src/ee/GPUetc/common/GNValue.h ../../src/ee/GPUetc/common/GTupleSchema.h ../../src/ee/GPUetc/expressions/Gcomparisonexpression.h ../../src/ee/GPUetc/expressions/makeexpressiontree.h ../../src/ee/GPUetc/common/nodedata.h ../../src/ee/GPUetc/expressions/treeexpression.h ../../src/ee/executors/index_join_gpu.h ../../src/ee/executors/join_gpu.h ../../src/ee/executors/ghash.h ../../src/ee/executors/gcommon/gpu_common.h ../../src/ee/GPUetc/storage/gtable.h ../../src/ee/GPUetc/indexes/TreeIndex.h ../../src/ee/GPUetc/indexes/KeyIndex.h ../../src/ee/GPUetc/storage/gtable.h"
    
#scan.cu ,join_gpu.cu ,hjoin_gpu.cu ,partitioning.cu, index_join_gpu.cu
#build library objects TreeIndex.co, gcommon.co, gtable.co
    makefile.write("objects/GPUetc/indexes/TreeIndex.o:../../src/ee/GPUetc/indexes/TreeIndex.cu %s\n"%GPUINC)
    makefile.write("\tnvcc $(INCLUDE) $(GPUFLAGS) $(GPUARCHFLAGS) --ptxas-options=-v  -o objects/GPUetc/indexes/TreeIndex.o -dc ../../src/ee/GPUetc/indexes/TreeIndex.cu\n")
    makefile.write("objects/GPUetc/indexes/TreeIndex.co: objects/GPUetc/indexes/TreeIndex.o %s\n"%GPUINC)
    makefile.write("\tnvcc $(INCLUDE) $(GPUFLAGS) $(GPUARCHFLAGS) -Xcompiler '-fPIC' --ptxas-options=-v -o objects/GPUetc/indexes/TreeIndex.co --device-link objects/GPUetc/indexes/TreeIndex.o\n")
    makefile.write("objects/GPUetc/storage/gtable.co: ../../src/ee/GPUetc/storage/gtable.cu %s\n"%GPUINC)
    makefile.write("\tnvcc $(INCLUDE) $(GPUFLAGS) $(GPUARCHFLAGS) -Xcompiler '-fPIC' --ptxas-options=-v  -o objects/GPUetc/storage/gtable.co -c ../../src/ee/GPUetc/indexes/gtable.cu\n")

#build join_gpu.co, index_join_gpu.co, hash_join.co    
    makefile.write("objects/executors/index_join_gpu.co:../../src/ee/executors/index_join_gpu.cu %s\n"%GPUINC)
    makefile.write("\tnvcc $(INCLUDE) $(GPUFLAGS) $(GPUARCHFLAGS) -Xcompiler '-fPIC' --ptxas-options=-v -o objects/executors/index_join_gpu.co -c %sindex_join_gpu.cu\n"%(GPUPATH))
    makefile.write("objects/executors/join_gpu.co:../../src/ee/executors/join_gpu.cu %s\n"%GPUINC)
    makefile.write("\tnvcc $(INCLUDE) $(GPUFLAGS) $(GPUARCHFLAGS) -Xcompiler '-fPIC' --ptxas-options=-v -o objects/executors/join_gpu.co -c %sjoin_gpu.cu\n"%(GPUPATH))
    makefile.write("objects/executors/ghash.co:../../src/ee/executors/ghash.cu %s\n"%GPUINC)
    makefile.write("\tnvcc $(INCLUDE) $(GPUFLAGS) $(GPUARCHFLAGS) -Xcompiler '-fPIC' --ptxas-options=-v -o objects/executors/ghash.co -c %sghash.cu\n"%(GPUPATH))
    makefile.write("objects/executors/gcommon.co:../../src/ee/executors/gcommon/gpu_common.cu %s\n"%GPUINC)
    makefile.write("\tnvcc $(INCLUDE) $(GPUFLAGS) $(GPUARCHFLAGS) -Xcompiler '-fPIC' --ptxas-options=-v -o objects/executors/gcommon.co -c %s/gcommon/gpu_common.cu\n"%(GPUPATH))




#    makefile.write("\tnvcc $(INCLUDE) $(GPUFLAGS) $(GPUARCHFLAGS) -cubin --ptxas-options=-preserve-relocs -o /home/anh/VoltDBdebug/index_join_gpu.cubin %sindex_join_gpu.cu\n"%(GPUPATH))
#    makefile.write("\tnvcc $(INCLUDE) $(GPUFLAGS) $(GPUARCHFLAGS) -G --source-in-ptx -ptx -o /home/anh/VoltDBdebug/mix_index_join_gpu.txt %sindex_join_gpu.cu\n"%(GPUPATH))



#scan_main.cpp ,GPUNIJ.cpp ,GPUSHJ.cpp, GPUIJ.cpp
    #makefile.write("objects/executors/scan_main.co:../../src/ee/executors/scan_main.cpp %s\n"%GPUINC)    
    #makefile.write("\tg++ $(INCLUDE) $(GPUFLAGS) -fPIC -o objects/executors/scan_main.co -c %sscan_main.cpp\n"%(GPUPATH))
#makefile.write("objects/executors/GPUNIJ.co:../../src/ee/executors/GPUNIJ.cpp objects/executors/join_gpu.cubin objects/executors/scan.co %s\n"%GPUINC)
    makefile.write("objects/executors/GPUNIJ.co:../../src/ee/executors/GPUNIJ.cpp objects/executors/join_gpu.co %s\n"%GPUINC)
    makefile.write("\tg++ $(INCLUDE) $(GPUFLAGS) -fPIC -o objects/executors/GPUNIJ.co -c %sGPUNIJ.cpp\n"%(GPUPATH))
    makefile.write("objects/executors/GPUSHJ.co:../../src/ee/executors/GPUSHJ.cpp objects/executors/hjoin_gpu.cubin objects/executors/partitioning.cubin objects/executors/scan.co %s\n"%GPUINC)
    makefile.write("\tg++ $(INCLUDE) $(GPUFLAGS) -fPIC -o objects/executors/GPUSHJ.co -c %sGPUSHJ.cpp\n"%(GPUPATH))
    makefile.write("objects/executors/GPUIJ.co:../../src/ee/executors/GPUIJ.cpp objects/executors/index_join_gpu.co objects/executors/gcommon.co objects/GPUetc/indexes/TreeIndex.co%s\n"%GPUINC)
    makefile.write("\tg++ $(INCLUDE) $(GPUFLAGS) -fPIC -o objects/executors/GPUIJ.co -c %sGPUIJ.cpp\n"%(GPUPATH))
    makefile.write("objects/executors/GPUINSERT.co:../../src/ee/executors/GPUINSERT.cpp %s\n"%GPUINC)
    makefile.write("\tg++ $(INCLUDE) $(GPUFLAGS) -fPIC -o objects/executors/GPUINSERT.co -c %sGPUINSERT.cpp\n"%(GPUPATH))    
    makefile.write("objects/executors/GPUHJ.co:../../src/ee/executors/GPUHJ.cpp objects/executors/ghash.co objects/executors/gcommon.co %s\n"%GPUINC)
    makefile.write("\tg++ $(INCLUDE) $(GPUFLAGS) -fPIC -o objects/executors/GPUHJ.co -c %sGPUHJ.cpp\n"%(GPUPATH))
#    makefile.write("objects/executors/GPUIJ.co: objects/executors/GPUIJ.o objects/executors/index_join_gpu.co %s\n"%GPUINC)
#    makefile.write("\tg++ $(INCLUDE) $(GPUFLAGS) -fPIC -o objects/executors/GPUIJ.co objects/executors/index_join_gpu.co objects/executors/GPUIJ.o\n")

#add part end

    LOCALTESTCPPFLAGS = LOCALCPPFLAGS + " -I%s" % (TEST_PREFIX)
    allsources = []
    for filename in input_paths:
        allsources += [(filename, LOCALCPPFLAGS, IGNORE_SYS_PREFIXES)]
    for filename in third_party_input_paths:
        allsources += [(filename, LOCALCPPFLAGS, IGNORE_SYS_PREFIXES)]
    for test in tests:
        binname, objectname, sourcename = namesForTestCode(test)
        allsources += [(sourcename, LOCALTESTCPPFLAGS, IGNORE_SYS_PREFIXES)]
    deps = getAllDependencies(allsources, 1)


    for filename in input_paths:
#add part
        mydeps = deps[filename]
        mydeps = [x.replace(INPUT_PREFIX, "$(SRC)") for x in mydeps]
        jni_objname, static_objname = outputNamesForSource(filename)
        filename = filename.replace(INPUT_PREFIX, "$(SRC)")
        jni_targetpath = OUTPUT_PREFIX + "/" + "/".join(jni_objname.split("/")[:-1])
        static_targetpath = OUTPUT_PREFIX + "/" + "/".join(static_objname.split("/")[:-1])
        os.system("mkdir -p %s" % (jni_targetpath))
        os.system("mkdir -p %s" % (static_targetpath))
        makefile.write(jni_objname + ": " + filename + " " + " ".join(mydeps) + GPUINC + "\n")
        makefile.write("\t$(CCACHE) $(COMPILE.cpp) $(GPUFLAGS) %s -o $@ %s\n" % (CTX.EXTRAFLAGS, filename))
        makefile.write(static_objname + ": " + filename + " " + " ".join(mydeps) + GPUINC + "\n")
        makefile.write("\t$(CCACHE) $(COMPILE.cpp) $(GPUFLAGS) %s -o $@ %s\n" % (CTX.EXTRAFLAGS, filename))
        makefile.write("\n")
#add part end

    for filename in third_party_input_paths:
        mydeps = deps[filename]
        mydeps = [x.replace(THIRD_PARTY_INPUT_PREFIX, "$(THIRD_PARTY_SRC)") for x in mydeps]
        jni_objname, static_objname = outputNamesForSource(filename)
        filename = filename.replace(THIRD_PARTY_INPUT_PREFIX, "$(THIRD_PARTY_SRC)")
        jni_targetpath = OUTPUT_PREFIX + "/" + "/".join(jni_objname.split("/")[:-1])
        static_targetpath = OUTPUT_PREFIX + "/" + "/".join(static_objname.split("/")[:-1])
        os.system("mkdir -p %s" % (jni_targetpath))
        os.system("mkdir -p %s" % (static_targetpath))
        makefile.write(jni_objname + ": " + filename + " " + " ".join(mydeps) + "\n")
        makefile.write("\t$(CCACHE) $(COMPILE.cpp) %s -o $@ %s\n" % (CTX.EXTRAFLAGS, filename))
        makefile.write(static_objname + ": " + filename + " " + " ".join(mydeps) + "\n")
        makefile.write("\t$(CCACHE) $(COMPILE.cpp) %s -o $@ %s\n" % (CTX.EXTRAFLAGS, filename))
    makefile.write("\n")

    for test in tests:
        binname, objectname, sourcename = namesForTestCode(test)

        # build the object file
        makefile.write("%s: ../../%s" % (objectname, sourcename))
        mydeps = deps[sourcename]
        for dep in mydeps:
            makefile.write(" ../../%s" % (dep))
        makefile.write("\n")
        makefile.write("\t$(CCACHE) $(COMPILE.cpp) -I../../%s -o $@ ../../%s\n" % (TEST_PREFIX, sourcename))

        # link the test
        makefile.write("%s: %s objects/volt.a\n" % (binname, objectname))
        makefile.write("\t$(LINK.cpp) -o %s %s objects/volt.a\n" % (binname, objectname))
        targetpath = OUTPUT_PREFIX + "/" + "/".join(binname.split("/")[:-1])
        os.system("mkdir -p %s" % (targetpath))

        pysourcename = sourcename[:-3] + "py"
        if os.path.exists(pysourcename):
            shutil.copy(pysourcename, targetpath)

    makefile.write("\n")
    makefile.close()
    return True

def buildIPC(CTX):
    retval = os.system("make --directory=%s prod/voltdbipc -j4" % (CTX.OUTPUT_PREFIX))
    return retval

def runTests(CTX):
    failedTests = []

    retval = os.system("make --directory=%s test -j4" % (CTX.OUTPUT_PREFIX))
    if retval != 0:
        return -1
    TESTOBJECTS_DIR = os.environ['TEST_DIR']
    TEST_PREFIX = CTX.TEST_PREFIX.rstrip("/")
    OUTPUT_PREFIX = CTX.OUTPUT_PREFIX.rstrip("/")

    tests = []
    for dir in CTX.TESTS.keys():
        input = CTX.TESTS[dir].split()
        tests += [TEST_PREFIX + "/" + dir + "/" + x for x in input]
    successes = 0
    failures = 0
    noValgrindTests = [ "CompactionTest", "CopyOnWriteTest", "harness_test", "serializeio_test" ]
    for test in tests:
        binname, objectname, sourcename = namesForTestCode(test)
        targetpath = OUTPUT_PREFIX + "/" + binname
        retval = 0
        if test.endswith("CopyOnWriteTest") and CTX.LEVEL == "MEMCHECK_NOFREELIST":
            continue
        if os.path.exists(targetpath + ".py"):
            retval = os.system("/usr/bin/env python " + targetpath + ".py")
        else:
            isValgrindTest = True;
            for test in noValgrindTests:
                if targetpath.find(test) != -1:
                    isValgrindTest = False;
            if CTX.PLATFORM == "Linux" and isValgrindTest:
                process = Popen(executable="valgrind", args=["valgrind", "--leak-check=full", "--show-reachable=yes", "--error-exitcode=-1", targetpath], stderr=PIPE, bufsize=-1)
                #out = process.stdout.readlines()
                allHeapBlocksFreed = False
                out_err = process.stderr.readlines()
                retval = process.wait()
                for str in out_err:
                    if str.find("All heap blocks were freed") != -1:
                        allHeapBlocksFreed = True
                if not allHeapBlocksFreed:
                    print "Not all heap blocks were freed"
                    retval = -1
                if retval == -1:
                    for str in out_err:
                        print str
                sys.stdout.flush()
            else:
                retval = os.system(targetpath)
        if retval == 0:
            successes += 1
        else:
            failedTests += [binname]
            failures += 1
    print "==============================================================================="
    print "TESTING COMPLETE (PASSED: %d, FAILED: %d)" % (successes, failures)
    for test in failedTests:
        print "TEST: " + test + " in DIRECTORY: " + CTX.OUTPUT_PREFIX + " FAILED"
    if failures == 0:
        print "*** SUCCESS ***"
    else:
        print "!!! FAILURE !!!"
    print "==============================================================================="

    return failures

def getCompilerVersion():
    vinfo = output = Popen(["gcc", "-v"], stderr=PIPE).communicate()[1]
    # Apple now uses clang and has its own versioning system.
    # Not sure we do anything that cares which version of clang yet.
    # This is pretty dumb code that could be improved as we support more
    #  compilers and versions.
    if output.find('clang') != -1:
        # this is a hacky way to find the real clang version from the output of
        # gcc -v on the mac. The version is right after the string "based on LLVM ".
        token = "based on LLVM "
        pos = vinfo.find(token)
        if pos != -1:
            pos += len(token)
            vinfo = vinfo[pos:pos+3]
            vinfo = vinfo.split(".")
            return "clang", vinfo, vinfo, 0
        # if not the expected apple clang format
        # this probably needs to be adjusted for clang on linux or from source
        return "clang", 0, 0, 0
    else:
        vinfo = vinfo.strip().split("\n")
        vinfo = vinfo[-1]
        vinfo = vinfo.split()[2]
        vinfo = vinfo.split(".")
        return "gcc", int(vinfo[0]), int(vinfo[1]), int(vinfo[2])

# get the version of gcc and make it avaliable
compiler_name, compiler_major, compiler_minor, compiler_point = getCompilerVersion()

