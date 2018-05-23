import time
import os
import tkinter as tk

from multiprocessing import Process, Queue

from parameter import *
from core import train_model, use_model


class UI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        # Init UI elements
        self.do_blink = False
        self.title("SeqLite")
        self.resizable(0, 0)
        self.grid_columnconfigure(8, minsize=5)
        self.grid_rowconfigure(4, minsize=5)
        self.protocol("WM_DELETE_WINDOW", self.close_ui)
        # Row 0:
        self.label1 = tk.Label(self, heigh=2, width=10, text="Select Mode:")
        self.rabtn1 = tk.Radiobutton(self, width=10, text="train", variable=TRAIN, value=True, indicatoron=0,
                                command=self.click_train)
        self.rabtn2 = tk.Radiobutton(self, width=10, text="use", variable=TRAIN, value=False, indicatoron=0,
                                command=self.click_use)
        # Row 1:
        self.text1 = tk.Text(self, height=10, width=60, font=("Arial", 14))
        self.text1["state"] = tk.DISABLED  # Disable edit
        self.scro1 = tk.Scrollbar(self)
        self.scro1.grid(sticky=tk.NS)  # North-south scrollbar
        self.scro1.config(command=self.text1.yview)
        self.text1.config(yscrollcommand=self.scro1.set)
        # Row 2:
        self.inp_line = tk.StringVar()
        self.ent1 = tk.Entry(self, width=60, font=("Arial", 14), textvariable=self.inp_line)
        self.ent1.config(state=tk.DISABLED)
        self.button1 = tk.Button(self, text=' Send ', command=self.click_run)
        self.button1.config(state=tk.DISABLED)

        # Row 0:
        self.label1.grid(column=0, row=0, columnspan=2, padx=5)
        self.rabtn1.grid(column=2, row=0, columnspan=2)
        self.rabtn2.grid(column=4, row=0, columnspan=2)
        # Row 1:
        self.text1.grid(column=0, row=1, columnspan=7, padx=5)
        self.scro1.grid(column=7, row=1)
        # Row 2:
        self.ent1.grid(column=0, row=2, columnspan=7, padx=5)
        self.button1.grid(column=7, row=2)

        # Init process
        self.out_que = Queue()
        self.in_que = Queue()
        self.train_proc = Process(target=train_model, args=(self.out_que,))
        self.use_proc = Process(target=use_model, args=(self.in_que, self.out_que))

    def click_train(self):
        self.ent1.config(state=tk.DISABLED)
        self.button1.config(state=tk.DISABLED)
        if not self.train_proc.is_alive():
            self.text1["state"] = tk.NORMAL
            self.text1.delete(1.0, tk.END)
            self.text1["state"] = tk.DISABLED
            if self.use_proc.is_alive():
                self.use_proc.terminate()
                # Reinit use_proc
                self.use_proc = Process(target=use_model, args=(self.in_que, self.out_que))
            self.train_proc.start()
            self.do_blink = True
            self.show_msg()

    def click_use(self):
        if self.train_proc.is_alive():
            self.train_proc.terminate()
            # Reinit train_proc
            self.train_proc = Process(target=train_model, args=(self.out_que,))
        self.do_blink = False
        self.text1["state"] = tk.NORMAL
        self.text1.delete(1.0, tk.END)
        self.text1["state"] = tk.DISABLED
        self.ent1.config(state=tk.NORMAL)
        self.button1.config(state=tk.NORMAL)
        self.do_blink = True
        self.show_msg()

    def click_run(self):
        if self.use_proc.is_alive():
            self.button1.config(text=' Wait ')
            time.sleep(0.2)
            self.button1.config(text=' Send ')
            return
        else:
            inp_line = self.inp_line.get()
            self.in_que.put(inp_line)
            self.text1["state"] = tk.NORMAL
            self.text1.insert(tk.INSERT, ' ' + inp_line + '\n')
            self.text1["state"] = tk.DISABLED
            self.ent1.delete(0, tk.END)
            self.use_proc.start()
            self.use_proc = Process(target=use_model, args=(self.in_que, self.out_que))

    def show_msg(self):
        if self.do_blink:
            if not self.out_que.empty():
                msg = self.out_que.get()
                self.text1["state"] = tk.NORMAL
                if msg.find('[EOS]') > -1:
                    msg = msg.replace('[EOS]', '')
                    msg = '>> ' + msg
                self.text1.insert(tk.INSERT, msg + '\n')
                self.text1["state"] = tk.DISABLED
            self.after(50, self.show_msg)
            """
            try:
                msg = self.out_que.get()
                self.text1["state"] = tk.NORMAL
                self.text1.insert(tk.INSERT, msg + '\n')
                self.text1["state"] = tk.DISABLED
                self.after(50, self.show_msg)
            except self.out_que.empty():
                self.after(50, self.show_msg)
            """

    def close_ui(self):
        if self.train_proc.is_alive():
            self.train_proc.terminate()
        if self.use_proc.is_alive():
            self.use_proc.terminate()
        self.destroy()


if __name__ == "__main__":
    # Init UI elements
    ui = UI()
    ui.mainloop()
