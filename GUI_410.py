import urwid
import os
import sys
import subprocess

def main():
    text_header = (u"Amazon Product Review Summarization & Analysis -  F8 exits.")
    text_intro = [ u"Hi! This tool will helps customer summarize and analyze reviews from Amazon product page,"
                   u"including review text summarization, sentiment analysis, key feature extraction. Finally, it will generate a markdown format report.",
                  ('important', u"\nHow to use"),
                  ('important', u"\nInput: "),
                    u"Amazon product url copied from browser.",
                  ('important', u"\nOutput: "),
                    u"A markdown format report under path: /output/product_asin (Asin is the id of an amazon product)"]
    input_box_intro = u"Input / Paste Amazon Product URL:"

    text_edit_alignments = input_box_intro
    text_edit_left = u""
    blank = urwid.Divider()

    def button_press(button):
        cur_text = text_edit_left
        frame.footer = urwid.Text(cur_text)

    input_box = urwid.Edit("", text_edit_left, align='left')

    listbox_content = [
                        blank,
                        urwid.Padding(urwid.Text(text_intro), left=2, right=2, min_width=20),
                        urwid.Text(text_edit_alignments),
                        urwid.AttrWrap(input_box,'editbx', 'editfc'),
                        ]
    header = urwid.AttrWrap(urwid.Text(text_header, align='center'), 'header')
    listbox = urwid.ListBox(urwid.SimpleListWalker(listbox_content))

    palette = [
        ('body','black','light gray', 'standout'),
        ('reverse','light gray','black'),
        ('header','white','dark red', 'bold'),
        ('important','dark blue','light gray',('standout','underline')),
        ('editfc','white', 'dark blue', 'bold'),
        ('editbx','light gray', 'dark blue'),
        ('editcp','black','light gray', 'standout'),
        ('bright','dark gray','light gray', ('bold','standout')),
        ('buttn','black','dark cyan'),
        ('buttnf','white','dark blue','bold'),
        ]

    output_widget = urwid.Text("Current Status:\n" )

    frame = urwid.Frame(body = urwid.AttrWrap(listbox, 'body'), header=header,
                        footer=output_widget)

    screen = urwid.raw_display.Screen()

    def received_output(data):
        output_widget.set_text(output_widget.text + data.decode('utf8'))

    proc = None
    def unhandled(key):
        if key == 'f8':
            raise urwid.ExitMainLoop()

        elif key == 'enter':
            url = input_box.get_edit_text()
            global proc
            proc = subprocess.Popen(['python', '-u', run_me, url], stdout=write_fd, close_fds=True)

    loop = urwid.MainLoop(frame, palette, screen, unhandled_input=unhandled)
    run_me = os.path.join(os.path.dirname(sys.argv[0]), 'Main.py')
    write_fd = loop.watch_pipe(received_output)
    loop.run()
    proc.kill()
if __name__ == '__main__':
    main()