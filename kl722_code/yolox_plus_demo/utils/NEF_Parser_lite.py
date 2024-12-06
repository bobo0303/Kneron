
import struct
import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

__version__ = "NEF Parser v0.0.1"

class ModelBin(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsModelBin(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ModelBin()
        x.Init(buf, n + offset)
        return x

    # ModelBin
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ModelBin
    def FwInfoAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

    # ModelBin
    def AllModelsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint8Flags, o)
        return 0

class SchemaVersion(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSchemaVersion(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SchemaVersion()
        x.Init(buf, n + offset)
        return x

    # SchemaVersion
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SchemaVersion
    def MajorNum(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # SchemaVersion
    def MinorNum(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 9

    # SchemaVersion
    def RevisionNum(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 1

class NEFHeader(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsNEFHeader(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NEFHeader()
        x.Init(buf, n + offset)
        return x

    # NEFHeader
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # NEFHeader
    def Target(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # NEFHeader
    def ToolchainVersion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # NEFHeader
    def SchemaVersion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = SchemaVersion()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

class NEFContent(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsNEFContent(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NEFContent()
        x.Init(buf, n + offset)
        return x

    # NEFContent
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # NEFContent
    def Header(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = NEFHeader()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # NEFContent
    def ModelBin(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            obj = ModelBin()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def read_nef(filename):
    out_target = -1

    with open(filename, 'rb') as fp:
        print("======== NEF Info ========= \n")
        b = fp.read()
        buf = bytearray(b)
        nef = NEFContent.GetRootAsNEFContent(buf, 0)
        header = nef.Header()

        info = header.ToolchainVersion()
        print("Toolchain ver= %s " %(str(info, 'utf-8')))

        maj   = header.SchemaVersion().MajorNum()
        minor = header.SchemaVersion().MinorNum()
        rev   = header.SchemaVersion().RevisionNum()
        print("Schema ver   = v%d.%d.%d\n" %(maj, minor, rev))
        print("============================")

        model_bin = nef.ModelBin()
        fwinfo = model_bin.FwInfoAsNumpy()
        allmodels_data = model_bin.AllModelsAsNumpy()

    return int(header.Target()), fwinfo, allmodels_data

def parse_fwinfo(fw_info):

    struct_fmt = '=IIIIIIIIIIIIII'
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    bindata = fw_info[:4].tobytes()
    count = struct.unpack('I', bindata)[0]

    modeldata = []
    for x in range(count):
        bindata = fw_info[4 : 15*4].tobytes()
        s = struct_unpack(bindata)
        modeldata.append(s)

    return count, modeldata

def align_number(number, base):
    return (number+base-1) & ~(base-1)

def extract_radix(fw_info, allmodels_data):

    model_cnt, m_data = parse_fwinfo(fw_info)
    assert model_cnt < 2, "[Error] This example not support multiple model NEF"
    model_num = 0 # only support single model for now
    offset = align_number(m_data[model_num][9], 16) + align_number(m_data[model_num][11], 16)

    struct_fmt = '=IIIIIIIIIIIIIII'
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    struct_netinput_node_fmt = '=IIIIIIIIII' 
    struct_netinput_node_len = struct.calcsize(struct_netinput_node_fmt)
    struct_netinput_node_unpack = struct.Struct(struct_netinput_node_fmt).unpack_from

    ret_in = []
    ret_innode_radix = 0

    bindata = allmodels_data[offset : offset + struct_len]
    raw = struct_unpack(bindata)
    assert raw[0] == 0x8EB5A462, "Invalid setup.bin format"

    ret_in_count = raw[13]
    if ret_in_count != 1:
        print("[Error] This Example not support multiple input")

    bindata = allmodels_data[offset + struct_len : offset + struct_len + struct_netinput_node_len]
    raw = struct_netinput_node_unpack(bindata)
    ret_innode_radix = raw[8]

    return ret_innode_radix

def extract_input_radix_from_kl720_nef(model_path):
    platform, fw_info, allmodels_data = read_nef(model_path)

    assert platform == 1, "[Error] Require 720 NEF model"

    innode_radix = extract_radix(fw_info, allmodels_data)

    return innode_radix
