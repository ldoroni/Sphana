# See https://aka.ms/customizecontainer to learn how to customize your debug container and how Visual Studio uses this Dockerfile to build your images for faster debugging.

# These ARGs allow for swapping out the base used to make the final image when debugging from VS
ARG LAUNCHING_FROM_VS
# This sets the base image for final, but only if LAUNCHING_FROM_VS has been defined
ARG FINAL_BASE_IMAGE=${LAUNCHING_FROM_VS:+aotdebug}

# This stage is used when running from VS in fast mode (Default for Debug configuration)
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS base
USER $APP_UID
WORKDIR /app
EXPOSE 8080
EXPOSE 8081

# This stage is used to build the service project
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
# Install clang/zlib1g-dev dependencies for publishing to native
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    clang zlib1g-dev
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["Sphana.Database/Sphana.Database.csproj", "Sphana.Database/"]
RUN dotnet restore "./Sphana.Database/Sphana.Database.csproj"
COPY . .
WORKDIR "/src/Sphana.Database"
RUN dotnet build "./Sphana.Database.csproj" -c $BUILD_CONFIGURATION -o /app/build

# This stage is used to publish the service project to be copied to the final stage
FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "./Sphana.Database.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=true

# This stage is used as the base for the final stage when launching from VS to support debugging in regular mode (Default when not using the Debug configuration)
FROM base AS aotdebug
USER root
# Install GDB to support native debugging
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gdb
USER app

# This stage is used in production or when running from VS in regular mode (Default when not using the Debug configuration)
FROM ${FINAL_BASE_IMAGE:-mcr.microsoft.com/dotnet/runtime-deps:10.0} AS final
WORKDIR /app
EXPOSE 8080
EXPOSE 8081
COPY --from=publish /app/publish .
COPY models ./models
ENTRYPOINT ["./Sphana.Database"]




# # Sphana Database - Neural RAG Database System
# # Multi-stage Dockerfile with GPU support

# # Stage 1: Build stage
# FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
# WORKDIR /src

# # Copy project files
# COPY ["Sphana.Database.Protos/Sphana.Database.Protos.csproj", "Sphana.Database.Protos/"]
# COPY ["Sphana.Database/Sphana.Database.csproj", "Sphana.Database/"]

# # Restore dependencies
# RUN dotnet restore "Sphana.Database/Sphana.Database.csproj"

# # Copy source code
# COPY . .

# # Build the application
# WORKDIR "/src/Sphana.Database"
# RUN dotnet build "Sphana.Database.csproj" -c Release -o /app/build

# # Stage 2: Publish stage
# FROM build AS publish
# RUN dotnet publish "Sphana.Database.csproj" -c Release -o /app/publish /p:UseAppHost=false

# # Stage 3: Runtime stage with CUDA support
# FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS final

# # Install CUDA dependencies (for GPU support)
# # Note: For production, use nvidia/cuda base image or install CUDA runtime
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     wget \
#     ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # Copy published application
# COPY --from=publish /app/publish .

# # Create directories for data persistence
# RUN mkdir -p /app/data/vector_index \
#     && mkdir -p /app/data/knowledge_graph \
#     && mkdir -p /app/data/properties \
#     && mkdir -p /app/models

# # Expose ports
# EXPOSE 5001
# EXPOSE 5000

# # Health check
# HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
#     CMD wget --no-verbose --tries=1 --spider http://localhost:5000/health || exit 1

# # Set environment variables
# ENV ASPNETCORE_URLS=http://+:5000;https://+:5001
# ENV ASPNETCORE_ENVIRONMENT=Production

# # Run as non-root user
# USER $APP_UID

# ENTRYPOINT ["dotnet", "Sphana.Database.dll"]