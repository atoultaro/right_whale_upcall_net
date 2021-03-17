import struct,math
from datetime import datetime,timedelta

class XWAVhdr:
    
    def __init__(self,filename):
        self._filename = filename
        self.xhd = {}
        self.raw = {}
        with open(filename,'rb') as f:
            self.xhd["ChunkID"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8") 
            self.xhd["ChunkSize"] = struct.unpack("<I",f.read(4))[0]
            self.xhd["Format"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8") 
            self.xhd["fSubchunkID"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8") 
            self.xhd["fSubchunkSize"] = struct.unpack("<I",f.read(4))[0]
            self.xhd["AudioFormat"] = struct.unpack("<H",f.read(2))[0]
            self.xhd["NumChannels"] = struct.unpack("<H",f.read(2))[0]
            self.xhd["SampleRate"] = struct.unpack("<I",f.read(4))[0]
            self.xhd["ByteRate"] = struct.unpack("<I",f.read(4))[0]
            self.xhd["BlockAlign"] = struct.unpack("<H",f.read(2))[0]
            self.xhd["BitsPerSample"] = struct.unpack("<H",f.read(2))[0]
            self.nBits = self.xhd["BitsPerSample"]
            self.samp = {}
            self.samp["byte"] = math.floor(self.nBits/8)
            self.xhd["hSubchunkID"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8") 
            self.xhd["hSubchunkSize"] = struct.unpack("<I",f.read(4))[0]
            self.xhd["WavVersionNumber"] = struct.unpack("<B",f.read(1))[0]
            self.xhd["FirmwareVersionNumber"] = struct.unpack("<10s",f.read(10))[0].decode("utf-8") 
            self.xhd["InstrumentID"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8")
            self.xhd["SiteName"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8") 
            self.xhd["ExperimentName"] = struct.unpack("<8s",f.read(8))[0].decode("utf-8") 
            self.xhd["DiskSequenceNumber"] = struct.unpack("<B",f.read(1))[0]
            self.xhd["DiskSerialNumber"] = struct.unpack("<8s",f.read(8))[0]
            self.xhd["NumOfRawFiles"] = struct.unpack("<H",f.read(2))[0]
            self.xhd["Longitude"] = struct.unpack("<i",f.read(4))[0]
            self.xhd["Latitude"] = struct.unpack("<i",f.read(4))[0]
            self.xhd["Depth"] = struct.unpack("<h",f.read(2))[0]
            self.xhd["Reserved"] = struct.unpack("<8s",f.read(8))[0]
            #Setup raw file information
            self.xhd["year"] = []
            self.xhd["month"] = []
            self.xhd["day"] = []
            self.xhd["hour"] = []
            self.xhd["minute"] = []
            self.xhd["secs"] = []
            self.xhd["ticks"] = []
            self.xhd["byte_loc"] = []
            self.xhd["byte_length"] = []
            self.xhd["write_length"] = []
            self.xhd["sample_rate"] = []
            self.xhd["gain"] = []
            self.xhd["padding"] = []
            self.raw["dnumStart"] = []
            #self.raw["dvecStart"] = []
            self.raw["dnumEnd"] = []
            #self.raw["dvecEnd"] = []
            for i in range(0,self.xhd["NumOfRawFiles"]):
                self.xhd["year"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["month"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["day"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["hour"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["minute"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["secs"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["ticks"].append(struct.unpack("<H",f.read(2))[0])
                self.xhd["byte_loc"].append(struct.unpack("<I",f.read(4))[0])
                self.xhd["byte_length"].append(struct.unpack("<I",f.read(4))[0])
                self.xhd["write_length"].append(struct.unpack("<I",f.read(4))[0])
                self.xhd["sample_rate"].append(struct.unpack("<I",f.read(4))[0])
                self.xhd["gain"].append(struct.unpack("<B",f.read(1))[0])
                self.xhd["padding"].append(struct.unpack("<7s",f.read(7))[0])
            
                self.raw["dnumStart"].append(datetime(self.xhd["year"][i]+2000,self.xhd["month"][i],self.xhd["day"][i],self.xhd["hour"][i],self.xhd["minute"][i],self.xhd["secs"][i],self.xhd["ticks"][i]*1000))
                self.raw["dnumEnd"].append(self.raw["dnumStart"][i]  + timedelta(seconds=((self.xhd["byte_length"][i]-2)/self.xhd["ByteRate"])))
            
            self.xhd["dSubchunkID"] = struct.unpack("<4s",f.read(4))[0].decode("utf-8")
            self.xhd["dSubchunkSize"] = struct.unpack("<I",f.read(4))[0]
            self.dtimeStart = self.raw["dnumStart"][0]
            self.dtimeEnd = self.raw["dnumEnd"][-1]
            
            
            
            
            
            
            