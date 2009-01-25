# export LD_LIBRARY_PATH=/usr/local/cuda/lib/
CUDADIR :=/usr/local/cuda

TOPDIR :=.
SRCDIR :=$(TOPDIR)/src
OBJDIR :=$(TOPDIR)/bin/gcc

SRCS +=$(SRCDIR)/nv/nvGlutWidgets.cpp $(SRCDIR)/nv/nvGLWidgets.cpp $(SRCDIR)/nv/nvWidgets.cpp
SRCS +=$(SRCDIR)/smallpt.cu
SRCS +=$(SRCDIR)/gl_scene.cc
SRCS +=$(SRCDIR)/tokaspt.cc

# heh.
OBJS := $(SRCS)
OBJS := $(subst $(SRCDIR)/nv,$(OBJDIR), $(OBJS))
OBJS := $(subst $(SRCDIR),$(OBJDIR), $(OBJS))
OBJS := $(patsubst %.cc,%.o,  $(OBJS))
OBJS := $(patsubst %.cpp,%.o, $(OBJS))
OBJS := $(patsubst %.cu,%.o,  $(OBJS))

all: $(OBJDIR) tokaspt


CXXFLAGS :=-pipe -g -O3 -march=native -ffast-math -fomit-frame-pointer -fno-rtti -fno-exceptions
CXXFLAGS +=-Wall -Wextra -Wshadow -Wno-unused
CXXFLAGS +=-MMD
CXXFLAGS +=-DNDEBUG
# FreeGlut has trouble parsing Glut init strings but pretends it's fine. kludge.
CXXFLAGS +=-DUSE_FREEGLUT

CXXFLAGS +=-I$(SRCDIR) -I$(SRCDIR)/nv
CXXFLAGS +=-I$(CUDADIR)/include -I$(CUDADIR)/common/inc/
LDFLAGS  +=-L$(CUDADIR)/lib -L$(CUDADIR)/common/lib/linux
LDFLAGS  +=-lcutil -lcuda -lcudart -lGL -lGLU -lGLEW_x86_64 -lglut # snatched from cuda

## NVCC
# grabbed from cuda examples. quickfix.
# /usr/local/cuda/bin/nvcc    --compiler-options -fno-strict-aliasing  -I. -I/usr/local/cuda/include -I../../common/inc -DUNIX -O3   -o data/threadMigration.cubin -cubin threadMigration.cu
NVCC := $(CUDADIR)/bin/nvcc
NVCCFLAGS :=-DUNIX -O3 -I$(CUDADIR)/include -I$(CUDADIR)/common/inc/
NVCCFLAGS +=--ptxas-options=-v -Xopencc "-Wall,-Wno-unused,-Wno-implicit-function-declaration"
## NVCCFLAGS +=--use_fast_math -arch sm_11
NVCCNFLAGS_NATIVE :=-fno-strict-aliasing
#NVCCNFLAGS_NATIVE +=-fomit-frame-pointer -frename-registers -march=native
NVCCFLAGS +=$(foreach flag, $(NVCCNFLAGS_NATIVE), --compiler-options $(flag))


## rules
$(OBJDIR)/%.o : $(SRCDIR)/nv/%.cpp
	$(COMPILE.cc) $< -o $@
	
$(OBJDIR)/%.o : $(SRCDIR)/%.cc
	$(COMPILE.cc) $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

## targets
tokaspt: $(OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)

clean: 
	rm $(OBJS) $(OBJDIR)/*.d tokaspt

$(OBJDIR):
	mkdir -p $(OBJDIR)
	
	
-include $(OBJDIR)/*.d