"""Every Lua file of the mod compiles under LuaJIT (the game's VM): a syntax error, or a function past LuaJIT's 200 locals
(generator.lua's build() is at that limit), fails here instead of in the game. Needs lupa (pip install lupa); skipped
without it. Run: python earthng/lua_compile_test.py
"""
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(ROOT, 'mod')


def luajit():
    for name in ('luajit21', 'luajit20'):
        try:
            return __import__('lupa.' + name, fromlist=[name]).LuaRuntime()
        except ImportError:
            pass
    return None


def lua_files():
    for folder, _, names in os.walk(MOD):
        for name in sorted(names):
            if name.endswith('.lua'):
                yield os.path.join(folder, name)


@unittest.skipIf(luajit() is None, 'needs LuaJIT: pip install lupa')
class Compile(unittest.TestCase):
    def test_every_lua_file_compiles(self):
        L = luajit()
        check = L.eval('function(src, name) local f, err = loadstring(src, "@" .. name) return err end')
        files = list(lua_files())
        self.assertTrue(files, 'no Lua files under mod/')
        failed = []
        for path in files:
            with open(path, 'rb') as f:
                src = f.read()
            err = check(src, os.path.relpath(path, ROOT))
            if err:
                failed.append(str(err))
        self.assertEqual(failed, [], '\n'.join(failed))


if __name__ == '__main__':
    unittest.main()
