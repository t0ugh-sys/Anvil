"""
Example: Using logging system
"""

from loop_agent.logging import Logger, LogLevel, LogOutput, get_logger


def main():
    # Create a logger
    logger = Logger(
        name="my_agent",
        level=LogLevel.DEBUG,
        output=LogOutput.FILE,
        file_path="logs/agent.log",
    )
    
    # Log events
    logger.info("Agent starting", goal="analyze data")
    logger.log_step(1, "Initializing")
    logger.log_tool("read_file", True, file="data.csv")
    logger.log_tool("search", False, pattern="error")
    logger.error("Something went wrong", error_code=500)
    
    # Use global logger
    global_logger = get_logger(level="INFO")
    global_logger.info("Using global logger")
    
    print("Logging example complete!")


if __name__ == "__main__":
    main()
