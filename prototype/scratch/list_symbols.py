import MetaTrader5 as mt5

if not mt5.initialize():
    print(f"MT5 initialize failed: {mt5.last_error()}")
    exit()

symbols = mt5.symbols_get()
if symbols is None:
    print("No symbols found.")
else:
    print(f"Total symbols: {len(symbols)}")
    # Print first 50 symbols to see the pattern
    for s in symbols[:100]:
        print(s.name)

mt5.shutdown()
