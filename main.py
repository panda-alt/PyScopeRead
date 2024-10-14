import time
import scipy.fft as fft
import scipy.signal as signal
import pyvisa as visa
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import re


class Oscilloscope:
    def __init__(self, device_name):
        self.time_scale = 0.00002                        # 20us/div
        self.channel_scale = 0.05                        # 0.05V/Div
        self.ResourceManager = visa.ResourceManager()
        self.instrument = self.ResourceManager.open_resource(device_name)
        command_list = [":TRIGger:MODE EDGE",             # 边沿触发
                        ":TRIGger:SWEep NORMal",          # 一般触发
                        ":TRIGger:SOURce EXT",            # 外部触发
                        ":TRIGger:EDGE:SLOPe POSitive",   # 上升沿触发
                        ":CHANnel1:COUPling DC",          # 直流耦合
                        ":CHAN1:DISP ON",                 # 打开通道1
                        ":CHAN1:OFFSet 0V",               # 通道1偏移0V
                        ":CHAN1:SCAL {:f}V".format(self.channel_scale),
                        ":TIMebase:SCALe {:f}".format(self.time_scale),
                        ":TIMebase:OFFSet 0.00015",       # 左偏移150us
                        ":WAVeform:SOURce CHAN1",         # 设置当前要查询波形数据的信号源为通道一
                        ":WAVeform:MODE RAW",             # 设置读取内存波形数据
                        ":ACQuire:MEMory:DEPTh 7K",       # 存储深度7K
                        ":WAVeform:STARt 1",              # 从第1个点开始读取
                        ":WAVeform:STOP 7000",            # 读取到7000个点（读取全部数据）
                        ":WAVeform:FORMat BYTE",          # 波形数据的返回格式为单字节模式
                        ":WAVeform:POINts 7000"           # 共采集7000个点（一次读取全部数据）
                        ]
        for command_i in command_list:  # 初始化设置
            self.instrument.write(command_i)
            command_query = re.sub(r"\s+.+", "?", command_i)
            print(("Set: {:<28s} ,Get: {}".format(command_i,
                                                  self.instrument.query(command_query))).replace("\n", ""))

    def read_waveform_data(self):
        data_merge = np.array([])
        star_t = 1
        self.instrument.write(":WAVeform:MODE RAW")   # 更新内存数据和重置START
        while star_t > 0:                             # START为-1说明本帧数据已读取完
            self.instrument.write(":WAVeform:DATA?")  # 获取波形指令
            data = self.instrument.read_raw()         # 读取数据
            data_np = np.frombuffer(data,             # 转换到 ndarray
                                    dtype=np.uint8,   # 此处只用到相对大小，未进行电压值转换
                                    count=len(data))
            data_merge = np.concatenate((data_merge, data_np[6:-3]))
            star_t = int(self.instrument.query(":WAVeform:START?"))
            time.sleep(0.05)                           # 暂停一会儿，以免读太快示波器卡死
        voltage = data_merge * self.channel_scale * 8 / 255 - self.channel_scale * 4
        time_array = np.linspace(0, self.time_scale * 14, len(voltage))
        return time_array, voltage

    def plot_save_data(self):
        plt.style.use("one")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        data_save = []
        for i_data in tqdm.tqdm(range(200)):
            time_array, voltage = self.read_waveform_data()
            len_data = len(voltage)
            window = signal.windows.hann(len_data)
            amp = np.abs(fft.fft(voltage * window, norm="forward")) * 4  # 正负频段*2，窗函数幅值恢复*2

            freq = fft.fftfreq(len_data, 0.00002 * 14 / len_data)
            index_g30l300 = (freq > 30000) * (freq < 300000)             # 截取 30k~300k频段
            amp_save = amp[index_g30l300]
            data_save.append(amp_save)
            axes[0].cla()
            axes[0].plot(time_array*1.0E6, voltage*1000)
            axes[0].set_xlabel("Time/us")
            axes[0].set_ylabel("Voltage/mV")
            axes[1].cla()
            axes[1].plot(freq[index_g30l300] / 1000, amp_save*1000)
            axes[1].set_xlabel("Frequency/kHz")
            axes[1].set_ylabel("Amp/mV")
            plt.draw()
            plt.pause(0.01)
        data_save = np.array(data_save)
        np.save("waveform.npy", data_save)
        fig = plt.figure()
        f_mesh, index_mesh = np.meshgrid(freq[index_g30l300]/1000, np.arange(data_save.shape[0]))
        plt.contourf(f_mesh, index_mesh, data_save*1000)
        plt.xlabel("freq/Hz")
        plt.ylabel("index")
        plt.colorbar(label="Amp/mV")
        plt.show()


if __name__ == '__main__':
    my_osc = Oscilloscope("USB0::0x5656::0x0853::APO1423190111::INSTR")
    my_osc.plot_save_data()
