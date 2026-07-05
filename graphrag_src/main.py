import asyncio
import time
 
from GraphGenerator import GraphGenerator
 
 
generator = GraphGenerator(max_concurrent=8)
start = time.perf_counter()
asyncio.run(generator.generate_graph())
elapsed = time.perf_counter() - start
hours, remainder = divmod(elapsed, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Total wall time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")