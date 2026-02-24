# See https://aka.ms/customizecontainer to learn how to customize your debug container and how Visual Studio uses this Dockerfile to build your images for faster debugging.

# These ARGs allow for swapping out the base used to make the final image when debugging from VS
ARG LAUNCHING_FROM_VS
# This sets the base image for final, but only if LAUNCHING_FROM_VS has been defined
ARG FINAL_BASE_IMAGE=${LAUNCHING_FROM_VS:+aotdebug}

# This stage is used when running from VS in fast mode (Default for Debug configuration)
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS base

# Install native library dependencies required by librocksdb.so and libfaiss_c.so (needed for debug mode too)
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libsnappy1v5 liblz4-1 libzstd1 libbz2-1.0 \
       libopenblas0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER $APP_UID

WORKDIR /app
EXPOSE 8080

# Allow the Linux dynamic linker to find transitive native dependencies (e.g. libfaiss.so needed by libfaiss_c.so)
ENV LD_LIBRARY_PATH=/app/bin/Debug/net10.0/runtimes/linux-x64/native:/app/bin/Release/net10.0/runtimes/linux-x64/native


# This stage is used to build the service project
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
# Install clang/zlib1g-dev dependencies for publishing to native
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    clang zlib1g-dev
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["Services/Sphana.DataStore/Sphana.DataStore.csproj", "Services/Sphana.DataStore/"]
RUN dotnet restore "./Services/Sphana.DataStore/Sphana.DataStore.csproj"
COPY . .
WORKDIR "/src/Services/Sphana.DataStore"
RUN dotnet build "./Sphana.DataStore.csproj" -c $BUILD_CONFIGURATION -o /app/build

# This stage is used to publish the service project to be copied to the final stage
FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "./Sphana.DataStore.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=true

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

# Install native library dependencies required by librocksdb.so and libfaiss_c.so
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libsnappy1v5 liblz4-1 libzstd1 libbz2-1.0 \
       libopenblas0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER app

WORKDIR /app
EXPOSE 8080
COPY --from=publish /app/publish .

# Allow the Linux dynamic linker to find transitive native dependencies (e.g. libfaiss.so needed by libfaiss_c.so)
ENV LD_LIBRARY_PATH=/app/runtimes/linux-x64/native

ENTRYPOINT ["./Sphana.DataStore"]
