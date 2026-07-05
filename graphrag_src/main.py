import asyncio

from GraphGenerator import GraphGenerator


generator = GraphGenerator(max_concurrent=8)
asyncio.run(generator.generate_graph())