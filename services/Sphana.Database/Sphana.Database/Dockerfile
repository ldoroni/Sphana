# See https://aka.ms/customizecontainer to learn how to customize your debug container and how Visual Studio uses this Dockerfile to build your images for faster debugging.

# These ARGs allow for swapping out the base used to make the final image when debugging from VS
ARG LAUNCHING_FROM_VS
# This sets the base image for final, but only if LAUNCHING_FROM_VS has been defined
ARG FINAL_BASE_IMAGE=${LAUNCHING_FROM_VS:+aotdebug}

# Start with NVIDIA CUDA runtime image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS cuda-base

# Install cuDNN 9 manually
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    gnupg \
    libicu70 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    libcudnn9-cuda-12 \
    && rm cuda-keyring_1.0-1_all.deb \
    && apt-get remove -y gnupg \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install .NET 10 runtime on top of CUDA
RUN wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh \
    && chmod +x dotnet-install.sh \
    && ./dotnet-install.sh --channel 10.0 --runtime aspnetcore --install-dir /usr/share/dotnet \
    && ln -s /usr/share/dotnet/dotnet /usr/bin/dotnet \
    && rm dotnet-install.sh \
    && apt-get clean

# Create app user (matching Microsoft's images)
RUN groupadd -r app --gid=1654 && useradd -r -g app --uid=1654 app

# Set CUDA environment
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# This stage is used when running from VS in fast mode (Default for Debug configuration)
FROM cuda-base AS base
USER app
WORKDIR /app
EXPOSE 8080
EXPOSE 8081

# This stage is used to build the service project
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    clang zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
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
RUN dotnet publish "./Sphana.Database.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

# This stage is used as the base for the final stage when launching from VS to support debugging in regular mode
FROM base AS aotdebug
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gdb \
    && rm -rf /var/lib/apt/lists/*
USER app

# This stage is used in production or when running from VS in regular mode
FROM ${FINAL_BASE_IMAGE:-base} AS final
WORKDIR /app
EXPOSE 8080
EXPOSE 8081
COPY --from=publish /app/publish .
COPY models ./models
ENTRYPOINT ["dotnet", "Sphana.Database.dll"]
