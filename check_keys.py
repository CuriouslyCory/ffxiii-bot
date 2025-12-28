from pynput.keyboard import Key
print([k for k in dir(Key) if not k.startswith('_')])

